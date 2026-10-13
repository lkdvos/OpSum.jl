using Test
using OpSum
using OpSum: mpo, irrep_mpo_tensors, instantiate, couple, opsum, expterm, SiteOperator, storedpairs,
    _jordan_layout, _mpo_power, _mul, _bond_space, _deg_indices
using TensorKit
using TensorKit: @tensor
using SparseArrays: nnz

include(joinpath(@__DIR__, "testutils.jl"))   # plain

# The Kronecker power of an MPO tensor, built from structure constants alone: `mpo_tensormap` of the power MPO
# must be the dense `H^N`. Fermionic models are where a wrong braiding sign shows up.

dense(H) = plain(instantiate(H))
power_dense(P) = plain(mpo_tensormap(irrep_mpo_tensors(P.mpo)))

# models: `name => H` on a finite chain
function finite_models()
    S = spin_ops(Trivial)
    U = spin_ops(U1Irrep)
    SU = spin_ops(SU2Irrep)
    F1 = fermion_ops(U1Irrep)
    F0 = fermion_ops()
    Hb = hubbard_ops(U1Irrep, U1Irrep)
    chain(B, N) = FiniteChain(B, N)
    return [
        "Ising (Trivial) L=4" => opsum(
            chain(spin_basis(Trivial), 4),
            (-S.S_z_S_z[i, i + 1] for i in 1:3), (0.7 * S.S_x[i] for i in 1:4)
        ),
        "XXZ (U1) L=4" => opsum(
            chain(spin_basis(U1Irrep), 4),
            (0.5 * (U.S_plus_S_min[i, i + 1] + U.S_min_S_plus[i, i + 1]) + 0.8 * U.S_z_S_z[i, i + 1] for i in 1:3),
            (0.3 * U.S_z[i] for i in 1:4)
        ),
        "J1-J2 Heisenberg (U1) L=4" => opsum(
            chain(spin_basis(U1Irrep), 4),
            (0.5 * (U.S_plus_S_min[i, i + 1] + U.S_min_S_plus[i, i + 1]) + U.S_z_S_z[i, i + 1] for i in 1:3),
            (0.4 * (0.5 * (U.S_plus_S_min[i, i + 2] + U.S_min_S_plus[i, i + 2]) + U.S_z_S_z[i, i + 2]) for i in 1:2)
        ),
        "t-V fermions (U1) L=4" => opsum(
            chain(fermion_basis(U1Irrep), 4),
            (F1.f_plus_f_min[i, i + 1] + F1.f_plus_f_min[i + 1, i] + 2 * F1.f_num[i] * F1.f_num[i + 1] for i in 1:3)
        ),
        "t-V fermions (parity) L=4, next-nearest hop" => opsum(
            chain(fermion_basis(), 4),
            (F0.f_plus_f_min[i, i + 1] + F0.f_plus_f_min[i + 1, i] + 2 * F0.f_num[i] * F0.f_num[i + 1] for i in 1:3),
            (0.3 * (F0.f_plus_f_min[i, i + 2] + F0.f_plus_f_min[i + 2, i]) for i in 1:2)
        ),
        "pairing (parity) L=3" => opsum(
            chain(fermion_basis(), 3),
            (F0.f_plus_f_plus[i, i + 1] - F0.f_min_f_min[i, i + 1] + 0.5 * F0.f_num[i] for i in 1:2)
        ),
        "Hubbard (U1 x U1) L=3" => opsum(
            chain(hubbard_basis(U1Irrep, U1Irrep), 3),
            (
                Hb.u_plus_u_min[i, i + 1] + Hb.u_min_u_plus[i, i + 1] +
                    Hb.d_plus_d_min[i, i + 1] + Hb.d_min_d_plus[i, i + 1] for i in 1:2
            ),
            (4.0 * Hb.ud_num[i] for i in 1:3)
        ),
    ]
end

@testset "SU(2) is rejected" begin
    SU = spin_ops(SU2Irrep)
    H = opsum(FiniteChain(spin_basis(SU2Irrep), 3), (SU.S_exchange[i, i + 1] for i in 1:2))
    @test_throws ArgumentError _mpo_power(mpo(H), 2)
end

@testset "dense oracle: $name" for (name, H) in finite_models()
    W = mpo(H)
    D = dense(H)
    for N in 2:3
        P = _mpo_power(W, N)
        @test power_dense(P) ≈ D^N
    end
    @test power_dense(_mpo_power(W, 1)) ≈ D           # the Jordan form of W itself
end

@testset "levels, sectors and identity channels" begin
    H = Dict(finite_models())["t-V fermions (U1) L=4"]
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
    for (name, H) in finite_models()
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
        T = irrep_mpo_tensors(J.Ws, J.bondsectors[2:end], H.lattice)
        @test plain(mpo_tensormap(T)) ≈ dense(H)
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
                P = _mul(B, P, word[j])
                isempty(P) && break
            end
            isempty(P) && continue
            out[(ls, rs)] = P * sgn
        end
        (out, words)
    end
end

@testset "word memo: $name" for (name, H) in (
        (
            name => H for (name, H) in finite_models() if name in (
                    "Ising (Trivial) L=4", "XXZ (U1) L=4", "t-V fermions (U1) L=4", "Hubbard (U1 x U1) L=3",
                )
        )...,
    )
    W = mpo(H)
    for N in 2:3
        P = _mpo_power(W, N)
        ref = naive_power(W, N)
        @test P.stats.nmul == P.stats.nwords == sum(length(r[2]) for r in ref)
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
        @test P.stats.nwords < P.stats.nentries
    end
end

@testset "column pruning hook" begin
    S = spin_ops(Trivial)
    N = 4
    field = opsum(FiniteChain(spin_basis(Trivial), N), (0.7 * S.S_x[i] for i in 1:N))
    H = opsum(
        FiniteChain(spin_basis(Trivial), N),
        (-S.S_z_S_z[i, i + 1] for i in 1:(N - 1)), (0.7 * S.S_x[i] for i in 1:N)
    )
    W = mpo(H)
    # only the start and finish channels (Jordan indices 1 and 3): the powers of the field term alone
    only_ends(prefix) = all(l -> l != 2, prefix)
    for p in 2:3
        P = _mpo_power(W, p; keep = only_ends)
        @test all(lv -> all(t -> all(!=(2), t), lv), P.levels)
        @test all(lv -> length(lv) <= 2^p, P.levels)
        @test power_dense(P) ≈ dense(field)^p
    end
end

# Infinite chains: contract `k` unit cells with the start level on the left and the finish level on the right.
function select_left(T, sec, k)
    I = eltype(sec)
    sec[k] == unit(I) || error("not a neutral index")
    P = zeros(ComplexF64, Vect[I](unit(I) => 1) ← _bond_space(sec))
    block(P, unit(I))[1, _deg_indices(sec)[k]] = 1
    @tensor R[a o; i br] := P[a; bl] * T[bl o; i br]
    return R
end
function select_right(T, sec, k)
    I = eltype(sec)
    sec[k] == unit(I) || error("not a neutral index")
    Q = zeros(ComplexF64, _bond_space(sec) ← Vect[I](unit(I) => 1))
    block(Q, unit(I))[_deg_indices(sec)[k], 1] = 1
    @tensor R[bl o; i a] := T[bl o; i br] * Q[br; a]
    return R
end
function window_operator(Ts, secs, s, f, cells)
    L = length(Ts)
    window = repeat(Ts, cells)
    window[1] = select_left(window[1], secs[L], s)
    window[end] = select_right(window[end], secs[L], f)
    return plain(mpo_tensormap(window))
end

function infinite_models()
    U = spin_ops(U1Irrep)
    F1 = fermion_ops(U1Irrep)
    BU1 = spin_basis(U1Irrep)
    BF = fermion_basis(U1Irrep)
    xxz(i, j; Δ = 1.0) = 0.5 * (U.S_plus_S_min[i, j] + U.S_min_S_plus[i, j]) + Δ * U.S_z_S_z[i, j]
    tV(i, j; V = 2.0) = F1.f_plus_f_min[i, j] + F1.f_plus_f_min[j, i] + V * F1.f_num[i] * F1.f_num[j]
    return [
        "Heisenberg (U1)" => (opsum(InfiniteChain(BU1), xxz(1, 2)), 4),
        "XXZ + field, L=2 (U1)" => (
            opsum(InfiniteChain([BU1, BU1]), xxz(1, 2; Δ = 0.7), xxz(2, 3; Δ = 0.7), 0.3 * U.S_z[1]), 2,
        ),
        "t-V fermions (U1)" => (opsum(InfiniteChain(BF), tV(1, 2)), 4),
        "t-V fermions, L=2 (U1)" => (opsum(InfiniteChain([BF, BF]), tV(1, 2), tV(2, 3; V = 1.0)), 2),
        "decaying XXZ (U1)" => (opsum(InfiniteChain(BU1), expterm(xxz(1, 2); decay = 0.5)), 4),
        "decaying hopping + NN (fermions)" => (
            opsum(InfiniteChain(BF), tV(1, 2), expterm(F1.f_plus_f_min[1, 2] + F1.f_plus_f_min[2, 1]; decay = 0.4)), 4,
        ),
    ]
end

@testset "infinite chain window: $name" for (name, (H, cells)) in infinite_models()
    W = mpo(H)
    L = length(W)
    Ts = irrep_mpo_tensors(W)
    Hwin = window_operator(Ts, W.bondsectors, W.start[L], W.finish[L], cells)
    for N in 2:3
        P = _mpo_power(W, N)
        TP = irrep_mpo_tensors(P.mpo)
        @test P.mpo.start[L] == 1
        @test P.mpo.finish[L] == length(P.levels[L])
        Pwin = window_operator(TP, P.mpo.bondsectors, P.mpo.start[L], P.mpo.finish[L], cells)
        @test Pwin ≈ Hwin^N
    end
end
