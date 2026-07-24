# Differential tests for the persistent-graph MPO construction (irrepgraph.jl) against the
# transient-frontier sweep (_irrep_bipartite) and the SVD sweep (_irrep_svd). The public
# `irrep_mpo(…, BipartiteAlgorithm())` already routes to `_irrep_graph_bipartite`, so the whole
# existing suite exercises it; here we pin the *equivalence* to the reference implementations directly
# (same per-sector bond dims, same reconstructed operator) and the incremental suffix-merge on a
# larger chain.

using Test
using OpSum
using OpSum: irrep_mpo, irrep_mpo_tensors, mpo_terms, instantiate, TermSum, spin, couple
using OpSum: BipartiteAlgorithm
using OpSum: ITOTermTable, _irrep_bipartite, _irrep_graph_bipartite, _irrep_svd, _irrep_graph_svd
using OpSum: ITOGraph, _at_site!
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using TensorKit: @tensor
using LinearAlgebra: dot

LO(x) = OpSum.LocalOp(x)

# densify an assembled MPO / operator TensorMap to a matrix (drop dim-1 boundary/charge legs, keep the
# 2N physical axes, reorder to kron order) — the robust operator-equality check, unaffected by the
# per-bond basis rotation the SVD introduces (`mpo_terms` is not reliable there).
function physmatrix(t, N, d)
    A = convert(Array, t)
    A = dropdims(A; dims = Tuple(findall(==(1), size(A))))
    @assert ndims(A) == 2N
    perm = (reverse(1:N)..., reverse((N + 1):(2N))...)
    return reshape(permutedims(A, perm), d^N, d^N)
end
contract2(T) = @tensor Op[o1 o2 bL; bR i1 i2] := T[1][bL o1; i1 bm] * T[2][bm o2; i2 bR]
function contract3(T)
    return @tensor Op[o1 o2 o3 bL; bR i1 i2 i3] :=
        T[1][bL o1; i1 b1] * T[2][b1 o2; i2 b2] * T[3][b2 o3; i3 bR]
end
contractN(T) = length(T) == 2 ? contract2(T) : contract3(T)

# per-bond charge => multiplicity (order-independent; the per-sector basis choice / bond-index order
# may differ between implementations even when the compression is identical)
function _bondcounts(sec)
    d = Dict{Any, Int}()
    for c in sec
        d[c] = get(d, c, 0) + 1
    end
    return d
end
persector(secs) = map(_bondcounts, secs)

mpo_operator(Ws, secs, sites) = instantiate(mpo_terms(Ws, secs), sites)

# reference Hamiltonians spanning the non-abelian sectors in the suite (SU2, U1, trivial) and
# arities K ∈ {0,1,2,3}; each entry is (name, H, sites).
function reference_hamiltonians()
    cases = Tuple{String, TermSum, Vector}[]
    let V = SU2Space(1 // 2 => 1)
        for N in (2, 3, 4, 5, 6)
            H = reduce(+, dot(spin(V)[i], spin(V)[i + 1]) for i in 1:(N - 1))
            push!(cases, ("SU2 Heisenberg N=$N", H, fill(V, N)))
        end
        push!(cases, ("SU2 fields", spin(V)[1] + spin(V)[2] + spin(V)[3], fill(V, 3)))
        push!(
            cases, (
                "SU2 decoupled pairs N=6",
                dot(spin(V)[1], spin(V)[2]) + dot(spin(V)[3], spin(V)[4]) + dot(spin(V)[5], spin(V)[6]),
                fill(V, 6),
            )
        )
        S = spin(V)
        push!(
            cases, (
                "SU2 K=3 three-body",
                couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0)), fill(V, 3),
            )
        )
    end
    let V = Rep[U₁](0 => 1, 1 => 1)
        raise = LO(IrrepOperator(U1Irrep(1), 1))
        lower = LO(IrrepOperator(U1Irrep(-1), 1))
        for N in (3, 4)
            H = reduce(+, dot(raise[i], lower[i + 1]) for i in 1:(N - 1)) +
                reduce(+, dot(lower[i], raise[i + 1]) for i in 1:(N - 1))
            push!(cases, ("U1 hopping N=$N", H, fill(V, N)))
        end
        push!(
            cases, (
                "U1 K=3 three-body",
                couple(couple(raise[1], raise[2]; to = U1Irrep(2)), lower[3]; to = U1Irrep(1)), fill(V, 3),
            )
        )
    end
    let V = ℂ^2
        ops = instances(IrrepOperator, V)
        H = reduce(+, couple(LO(ops[2])[i], LO(ops[3])[i + 1]; to = unit(Trivial)) for i in 1:2)
        push!(cases, ("trivial sector N=3", H, fill(V, 3)))
    end
    return cases
end

@testset "graph VC ≡ transient-frontier bipartite" begin
    for (name, H, sites) in reference_hamiltonians()
        N = length(sites)
        tt = ITOTermTable(H, sites)
        Wo, so = _irrep_bipartite(tt, N)
        Wg, sg = _irrep_graph_bipartite(tt, N)
        @testset "$name" begin
            # identical per-sector bond dimensions at every bond
            @test persector(so) == persector(sg)
            # both reconstruct the original term-sum exactly (lossless), hence the same operator
            @test mpo_operator(Wg, sg, sites) ≈ mpo_operator(Wo, so, sites)
            back = mpo_terms(Wg, sg)
            @test Set(keys(back.terms)) == Set(keys(H.terms))
            @test all(back.terms[k] ≈ H.terms[k] for k in keys(H.terms))
        end
    end
end

@testset "public BipartiteAlgorithm selector uses the graph path" begin
    V = SU2Space(1 // 2 => 1)
    N = 4
    sites = fill(V, N)
    H = reduce(+, dot(spin(V)[i], spin(V)[i + 1]) for i in 1:(N - 1))
    Wsel, ssel = irrep_mpo(H, sites, BipartiteAlgorithm())
    Wg, sg = _irrep_graph_bipartite(ITOTermTable(H, sites), N)
    @test persector(ssel) == persector(sg)
    @test mpo_operator(Wsel, ssel, sites) ≈ mpo_operator(Wg, sg, sites)
    # tensor assembly is unchanged and materialises the same symmetric operator
    T = irrep_mpo_tensors(Wsel, ssel, sites)
    @test T isa Vector{<:AbstractTensorMap}
    @test length(T) == N
end

@testset "graph SVD (lossless) reconstructs the operator" begin
    # The persistent-graph SVD sweep (ITensor QR-backend port). Lossless it is at parity with
    # `_irrep_svd` (same represented operator); it is *not* the default `SVDBondAlgorithm` backend
    # because under truncation it implements sequential-sweep semantics rather than `_irrep_svd`'s
    # per-bond-independent truncation (see its docstring). The SVD rotates each bond's basis, so the
    # robust equality check is via the assembled TensorMaps contracted against the dense oracle — not
    # `mpo_terms` (which needs an intact identity backbone). Restrict to N ≤ 3 (contract2 / contract3).
    for (name, H, sites) in reference_hamiltonians()
        N = length(sites)
        N <= 3 || continue
        d = dim(sites[1])
        tt = ITOTermTable(H, sites)
        Wg, sg = _irrep_graph_svd(tt, N, nothing)
        # `physmatrix` drops the boundary/charge legs, so it needs a densifiable (dim-1 total charge)
        # operator; a net-charge Hamiltonian (e.g. the spin-1 field sum) is excluded here.
        all(c -> dim(c) == 1, sg[end]) || continue
        Mgraph = physmatrix(contractN(irrep_mpo_tensors(Wg, sg, sites)), N, d)
        Mexact = physmatrix(instantiate(H, sites), N, d)
        @testset "$name" begin
            @test Mgraph ≈ Mexact
        end
    end
end

@testset "graph SVD lossless bond dims match _irrep_svd" begin
    # Pin the docstring's parity claim on the internal bonds (1 … N-1). The right boundary (bond N) is
    # excluded because `_irrep_svd` hardcodes it to `unit(I)` whereas the graph SVD reports the true
    # total charge — they agree only for charge-0 Hamiltonians, so compare the internal bonds only.
    for (name, H, sites) in reference_hamiltonians()
        N = length(sites)
        N >= 2 || continue
        tt = ITOTermTable(H, sites)
        _, so = _irrep_svd(tt, N, nothing)
        _, sg = _irrep_graph_svd(tt, N, nothing)
        @testset "$name" begin
            @test persector(so[1:(N - 1)]) == persector(sg[1:(N - 1)])
        end
    end
end

@testset "incremental suffix-merge on a longer chain" begin
    # exercise many at_site! steps; the incremental (LCP-driven) suffix-merge must reproduce the
    # transient sweep's per-sector bond dims and stay lossless on an 8-site Heisenberg chain.
    V = SU2Space(1 // 2 => 1)
    N = 8
    sites = fill(V, N)
    H = reduce(+, dot(spin(V)[i], spin(V)[i + 1]) for i in 1:(N - 1))
    tt = ITOTermTable(H, sites)
    Wo, so = _irrep_bipartite(tt, N)
    Wg, sg = _irrep_graph_bipartite(tt, N)
    @test persector(so) == persector(sg)
    back = mpo_terms(Wg, sg)
    @test Set(keys(back.terms)) == Set(keys(H.terms))
    @test all(back.terms[k] ≈ H.terms[k] for k in keys(H.terms))
end

@testset "right vertices persist while their count shrinks" begin
    # the persistent graph keeps one right vertex per (surviving) suffix class; after each at_site!
    # the right-vertex count is non-increasing (suffix-merge only merges), ending at 1 (empty suffix).
    V = SU2Space(1 // 2 => 1)
    N = 5
    H = reduce(+, dot(spin(V)[i], spin(V)[i + 1]) for i in 1:(N - 1))
    g = ITOGraph(ITOTermTable(H, fill(V, N)), N)
    counts = Int[]
    for i in 1:N
        _at_site!(g, i)
        push!(counts, length(g.rrepr))
    end
    @test issorted(counts; rev = true)
    @test counts[end] == 1
end
