using Test
using OpSum
using OpSum: mpo, mpo_tensors, opsum, expterm, SiteOperator, storedpairs, _jordan_layout, _mpo_power, site_product
using TensorKit
using SparseArrays: nnz

include(joinpath(@__DIR__, "..", "testutils.jl"))

# The Kronecker power of an MPO tensor, built from structure constants alone: `mpo_tensormap` of the power MPO
# must be the dense `H^N`. Fermionic models are where a wrong braiding sign shows up.

power_dense(P) = plain(mpo_tensormap(mpo_tensors(P.mpo)))

# the bosonic oracle on Ising (N = 2, 3); the fermionic sign rule on the U1 fermions (N = 2, 3), the parity fermions with
# a next-nearest hop (N = 2) and Hubbard U1 x U1 (N = 2). Every further model only adds compilation.
@testset "dense oracle: $name" for (name, Ns) in (
        ("Ising (Trivial)", 2:3), ("t-V fermions (U1)", 2:3),
        ("t-V fermions (parity), next-nearest hop", 2:2), ("Hubbard (U1 x U1)", 2:2),
    )
    H = only(finite_models(name))[2]
    W = mpo(H)
    D = densify(H)
    for N in Ns
        P = _mpo_power(W, N)
        @test power_dense(P) ≈ D^N
    end
    name == "t-V fermions (U1)" && @test power_dense(_mpo_power(W, 1)) ≈ D   # the Jordan form of W itself
end

@testset "levels, sectors and identity channels" begin
    H = only(finite_models("t-V fermions (U1)"))[2]
    W = mpo(H)
    J = _jordan_layout(W)
    n = length.(J.bondsectors)                           # N + 1 bonds
    L = length(W)
    for N in 1:3
        P = _mpo_power(W, N)
        @test length(P.levels) == L
        for i in 1:L
            nb = n[i + 1]
            @test length(P.levels[i]) == nb^N
            @test P.levels[i][1] == ntuple(_ -> 1, N)
            i < L && @test P.levels[i][2] == ntuple(k -> k == 1 ? 2 : 1, N)   # layer 1 fastest
            # sectors are the fused charges of the components
            for (k, t) in enumerate(P.levels[i])
                @test P.mpo.bondsectors[i][k] == reduce((a, b) -> only(a ⊗ b), J.bondsectors[i + 1][collect(t)])
            end
            @test size(P.mpo.Ws[i]) == (i == 1 ? 1 : n[i]^N, nb^N)
        end
        @test P.mpo.start == [i < L ? 1 : 0 for i in 1:L]
        @test P.mpo.finish == [i < L ? n[i + 1]^N : 1 for i in 1:L]
        @test P.mpo.bondsectors[L] == [unit(sectortype(H.lattice))]
        # the power of a Jordan MPO is Jordan: identities at the corners
        for i in 1:L
            Wi = P.mpo.Ws[i]
            i < L && @test isone(Wi[1, 1])
            i > 1 && @test isone(Wi[size(Wi, 1), size(Wi, 2)])
        end
    end
end

@testset "jordan layout" begin
    for (name, H) in finite_models("Ising (Trivial)", "t-V fermions (U1)", "t-V fermions (parity), next-nearest hop")
        W = mpo(H)
        J = _jordan_layout(W)
        L = length(W)
        Ts = jordan_mpo_tensors(H)
        @test J.start == vcat([1], fill(1, L - 1), [0])
        @test J.finish[1] == 0 && J.finish[L + 1] == 1
        for i in 1:L
            @test size(J.Ws[i]) == (length(J.bondsectors[i]), length(J.bondsectors[i + 1]))
            @test size(Ts[i], 4) == length(J.bondsectors[i + 1])
            @test size(Ts[i], 1) == length(J.bondsectors[i])
        end
        for i in 2:L
            @test J.finish[i] == length(J.bondsectors[i])
            @test J.bondsectors[i][1] == J.bondsectors[i][end] == unit(sectortype(H.lattice))
        end
        # nothing instantiated, yet the same operator
        T = mpo_tensors(J.Ws, J.bondsectors[2:end], H.lattice)
        @test plain(mpo_tensormap(T)) ≈ densify(H)
    end
end

@testset "errors match jordan_mpo_tensors" begin
    S = spin_ops(U1Irrep)
    B = spin_basis(U1Irrep)
    h = 0.5 * (S.S_plus_S_min[1, 2] + S.S_min_S_plus[1, 2]) + S.S_z_S_z[1, 2]
    Hd = opsum(FiniteChain(B, 4), expterm(h; decay = 0.5))
    @test_throws ArgumentError jordan_mpo_tensors(Hd)
    @test_throws ArgumentError _jordan_layout(mpo(Hd))
    @test_throws ArgumentError _mpo_power(mpo(Hd), 2)
    @test_throws ArgumentError _mpo_power(mpo(Hd), 0)
    # a charged operator has no neutral right boundary
    Sp = matrixunit(B, U1Irrep(1 // 2), U1Irrep(-1 // 2))
    Hc = opsum(FiniteChain(B, 3), Sp[1] + Sp[2])
    @test_throws ArgumentError _jordan_layout(mpo(Hc))
end

# The memo: every distinct active word (per site) is multiplied once, and the memoised entries are those of the
# naive product of the layer entries.
function naive_power(W, N)
    J = _jordan_layout(W)
    L = length(W)
    return map(1:L) do i
        B = W.lattice[i]
        ent = [(Tuple(idx), op) for (idx, op) in storedpairs(J.Ws[i])]
        out = Dict{Any, SiteOperator}()
        words = Set{Vector{SiteOperator}}()
        for combo in Iterators.product(ntuple(_ -> ent, N)...)
            ls = ntuple(k -> combo[k][1][1], N)
            rs = ntuple(k -> combo[k][1][2], N)
            ops = SiteOperator[last(c) for c in combo]
            sgn = 1.0
            for k in 1:N, m in (k + 1):N
                q = only(J.bondsectors[i + 1][rs[k]] ⊗ dual(J.bondsectors[i][ls[k]]))
                sgn *= OpSum._braidsign(q, J.bondsectors[i + 1][rs[m]])
            end
            word = filter(!isone, ops)
            # distinct words of length >= 2 reached before a product vanishes
            P = isempty(word) ? one(SiteOperator{sectortype(B)}) : first(word)
            for j in 2:length(word)
                push!(words, word[1:j])
                P = site_product(B, P, word[j])
                isempty(P) && break
            end
            isempty(P) && continue
            out[(ls, rs)] = P * sgn
        end
        (out, words)
    end
end

@testset "word memo: $name" for (name, H) in
    finite_models("Ising (Trivial)", "t-V fermions (U1)", "Hubbard (U1 x U1)")
    W = mpo(H)
    for N in 2:3
        P = _mpo_power(W, N)
        ref = naive_power(W, N)
        @test P.stats.nmul == sum(length(r[2]) for r in ref)
        @test P.stats.nentries == sum(length(r[1]) for r in ref)
        # same entries as the naive fold
        for i in 1:length(W)
            rowlv = i == 1 ? [ntuple(_ -> 1, N)] : P.levels[i - 1]
            idx(lv) = Dict(t => k for (k, t) in enumerate(lv))
            ri, ci = idx(rowlv), idx(P.levels[i])
            M = P.mpo.Ws[i]
            for ((ls, rs), op) in ref[i][1]
                @test M[ri[ls], ci[rs]] ≈ op
            end
            @test nnz(M) == length(ref[i][1])
        end
        # sharing: fewer words than entries
        @test P.stats.nmul < P.stats.nentries
    end
end

@testset "infinite chain window: $name" for (name, (H, cells)) in infinite_models("t-V fermions (U1)")
    W = mpo(H)
    L = length(W)
    Ts = mpo_tensors(W)
    Hwin = window_operator(Ts, W.bondsectors, W.start[L], W.finish[L], cells)
    for N in 2:3
        P = _mpo_power(W, N)
        TP = mpo_tensors(P.mpo)
        @test P.mpo.start[L] == 1
        @test P.mpo.finish[L] == length(P.levels[L])
        Pwin = window_operator(TP, P.mpo.bondsectors, P.mpo.start[L], P.mpo.finish[L], cells)
        @test Pwin ≈ Hwin^N
    end
end
