# Differential tests for the persistent-graph MPO construction (irrepgraph.jl) against the
# transient-frontier sweep (_irrep_bipartite) and the SVD sweep (_irrep_svd). The public
# `irrep_mpo(…, BipartiteAlgorithm())` already routes to `_irrep_graph_bipartite`, so the whole
# existing suite exercises it; here we pin the *equivalence* to the reference implementations directly
# (same per-sector bond dims, same reconstructed operator) and the incremental suffix-merge on a
# larger chain.

using Test
using OpSum
using OpSum: irrep_mpo, irrep_mpo_tensors, mpo_terms, instantiate, TermSum, spin, couple, scalarop
using OpSum: BipartiteAlgorithm
using OpSum: ITOTermTable, _irrep_bipartite, _irrep_graph_bipartite, _irrep_svd, _irrep_graph_svd
using OpSum: ITOGraph, _at_site!
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using TensorKit: @tensor
using LinearAlgebra: dot, norm

include(joinpath(@__DIR__, "testutils.jl"))   # LO, physmatrix

# densify an assembled MPO / operator TensorMap to a matrix (drop dim-1 boundary/charge legs, keep the
# 2N physical axes, reorder to kron order) — the robust operator-equality check, unaffected by the
# per-bond basis rotation the SVD introduces (`mpo_terms` is not reliable there).
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
        # Coupling S₁⊗S₂ to charge 0 makes the running bond charge trivial, so from bond 2 on the
        # K=3 term is indistinguishable from a not-yet-started on-site S₄ — their suffix classes
        # genuinely merge. See the "pending suffix class can collide with a started one" testset.
        push!(
            cases, (
                "SU2 K=3 tail collides with on-site field",
                couple(couple(S[1], S[2]; to = SU2Irrep(0)), S[4]; to = SU2Irrep(1)) + 2.0 * S[4],
                fill(V, 4),
            )
        )
        # boundary shapes: nothing active at site 1; a term whose first active site is the last one
        push!(cases, ("SU2 no term starts at site 1", dot(S[2], S[3]), fill(V, 3)))
        # a term whose first active site is the last one (uniform total charge: both are single spins,
        # mixing total charges in one `TermSum` is out of scope for the sweeps)
        push!(cases, ("SU2 first active site == N", S[1] + S[4], fill(V, 4)))
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
        # A charge-0 letter leaves the running bond trivial, so a pending on-site term can share a
        # suffix class with a started two-site one — the U(1)/fermionic analogue of an `n̂`
        # chemical-potential term colliding with the tail of an `n̂ᵢn̂ⱼ` interaction.
        z = LO(IrrepOperator(U1Irrep(0), 1))
        push!(
            cases, (
                "U1 charge-0 on-site collides with two-site tail",
                couple(z[1], z[3]; to = U1Irrep(0)) + 0.5 * z[3], fill(V, 3),
            )
        )
    end
    let V = ℂ^2
        ops = instances(IrrepOperator, V)
        H = reduce(+, couple(LO(ops[2])[i], LO(ops[3])[i + 1]; to = unit(Trivial)) for i in 1:2)
        push!(cases, ("trivial sector N=3", H, fill(V, 3)))
        # In the trivial sector every bond charge is the unit, so the collision above degenerates to
        # "a pending term's whole content equals a started term's tail".
        push!(
            cases, (
                "trivial on-site collides with two-site tail",
                couple(LO(ops[2])[1], LO(ops[3])[3]; to = unit(Trivial)) + 0.5 * LO(ops[3])[3],
                fill(V, 3),
            )
        )
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

@testset "suffix-merge over many site steps" begin
    # exercise many at_site! steps; the per-site signature regroup must reproduce the transient
    # sweep's per-sector bond dims and stay lossless on an 8-site Heisenberg chain.
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

@testset "K=0 identity terms" begin
    # A K=0 term is all pass-through: its suffix class is the "done" class from bond 0 on, and its
    # left vertex coincides with the identity/start channel. Checked via per-sector bond dims and the
    # `mpo_terms` round-trip rather than the dense oracle, because `instantiate` cannot represent a
    # `TermSum` that mixes K=0 and K>0 terms (a pre-existing limitation of the oracle, not the sweep).
    V = SU2Space(1 // 2 => 1)
    S = spin(V)
    for (name, H, N) in (
            ("pure identity", scalarop(2.5, V)[1], 3),
            ("identity plus bond", scalarop(2.5, V)[1] + dot(S[1], S[2]), 3),
            ("identity plus two bonds", scalarop(-1.5, V)[1] + dot(S[1], S[2]) + dot(S[2], S[3]), 3),
        )
        sites = fill(V, N)
        tt = ITOTermTable(H, sites)
        Wo, so = _irrep_bipartite(tt, N)
        Wg, sg = _irrep_graph_bipartite(tt, N)
        @testset "$name" begin
            @test persector(so) == persector(sg)
            back = mpo_terms(Wg, sg)
            @test Set(keys(back.terms)) == Set(keys(H.terms))
            @test all(back.terms[k] ≈ H.terms[k] for k in keys(H.terms))
        end
    end
end

@testset "suffix signature needs the running bond charge" begin
    # A suffix class is named by `(interned remaining factors, running bond charge)`. The second
    # component is not decoration: two terms can have *identical* remaining factors yet different
    # accumulated charge, and then `_op_at_ito` fills the idle sites between with pass-throughs at
    # different charges — so the suffix paths differ and the classes must stay apart. Here both terms
    # end with the same spin-1 letter at site 4 but reach bond 2 at charge 0 and charge 1 respectively.
    # Dropping the charge from the signature merges them and trips the sector-purity assert.
    #
    # This case cannot go through `reference_hamiltonians`: `_irrep_bipartite` keys its suffix classes
    # on `_op_at_ito` alone (`_suffix_path`), so it omits the running charge and over-merges here,
    # tripping its own purity assert. The graph sweep is strictly more general in this regime, so the
    # check is against `mpo_terms` and the dense oracle rather than against the transient sweep.
    V = SU2Space(1 // 2 => 1)
    zl, sl = instances(IrrepOperator, V)             # charge-0 and charge-1 letters
    @test zl.c == SU2Irrep(0) && sl.c == SU2Irrep(1)
    z, s = LO(zl), LO(sl)
    N = 4
    sites = fill(V, N)
    H = couple(z[1], s[4]; to = SU2Irrep(1)) + 2.0 * couple(s[1], s[4]; to = SU2Irrep(1))
    tt = ITOTermTable(H, sites)

    # the two classes agree on the remaining factors at bond 2 but not on the running charge
    @test OpSum._op_at_ito(tt, 1, 4) == OpSum._op_at_ito(tt, 2, 4)
    @test OpSum._op_at_ito(tt, 1, 3) != OpSum._op_at_ito(tt, 2, 3)

    Wg, sg = _irrep_graph_bipartite(tt, N)
    back = mpo_terms(Wg, sg)
    @test Set(keys(back.terms)) == Set(keys(H.terms))
    @test all(back.terms[k] ≈ H.terms[k] for k in keys(H.terms))
    @test norm(instantiate(back, sites) - instantiate(H, sites)) < 1.0e-10
end

@testset "pending suffix class can collide with a started one" begin
    # The invariant that makes lazy right-vertex insertion subtle. `_op_at_ito` fills idle sites with
    # a pass-through carrying the *running* bond charge, so a started term whose accumulated charge
    # has fused back to the unit is indistinguishable, over its idle sites, from a term that has not
    # started yet. Here `A₁B₃` and the on-site `B₃` share the suffix class from site 2 on, and the
    # minimum vertex cover exploits it: it covers the single shared right vertex and both left
    # vertices fold their weighted letters into that one column, so every bond stays 1-dimensional.
    #
    # A lazy scheme that simply defers a pending term until its first active site loses this merge
    # and reports [2, 2, 1] instead — which is why the pending signatures have to be probed at every
    # site. No showcase model triggers this (all K=2, no on-site terms), so this is the guard.
    V = ℂ^2
    ops = instances(IrrepOperator, V)
    A, B = LO(ops[2]), LO(ops[3])
    N = 3
    sites = fill(V, N)
    H = couple(A[1], B[3]; to = unit(Trivial)) + 0.5 * B[3]
    tt = ITOTermTable(H, sites)

    # the two terms really are in the same suffix class at bond 1 (they differ only at site 1)
    @test OpSum._op_at_ito(tt, 1, 2) == OpSum._op_at_ito(tt, 2, 2)
    @test OpSum._op_at_ito(tt, 1, 3) == OpSum._op_at_ito(tt, 2, 3)
    @test OpSum._op_at_ito(tt, 1, 1) != OpSum._op_at_ito(tt, 2, 1)

    Wo, so = _irrep_bipartite(tt, N)
    Wg, sg = _irrep_graph_bipartite(tt, N)
    @test length.(so) == [1, 1, 1]
    @test length.(sg) == [1, 1, 1]
    back = mpo_terms(Wg, sg)
    @test Set(keys(back.terms)) == Set(keys(H.terms))
    @test all(back.terms[k] ≈ H.terms[k] for k in keys(H.terms))
end

@testset "live right vertices stay bounded independently of N" begin
    # The persistent graph keeps one right vertex per *reachable* suffix class. Under lazy insertion
    # the count is no longer monotone — terms enter at their first active site — so what matters is
    # the property that buys the linear scaling: for a finite-range model the live count is bounded by
    # a constant, not by the number of terms. (Eagerly it peaked at M, making the sweep Θ(N·M).)
    V = SU2Space(1 // 2 => 1)
    sizes = (5, 20, 80)
    livecounts = map(sizes) do N
        H = reduce(+, dot(spin(V)[i], spin(V)[i + 1]) for i in 1:(N - 1))
        g = ITOGraph(ITOTermTable(H, fill(V, N)), N)
        counts = Int[]
        for i in 1:N
            _at_site!(g, i)
            push!(counts, length(g.rrepr))
        end
        @test counts[end] == 1                 # only the exhausted ("done") suffix class survives
        # summed over the sweep the right-vertex visits are linear in N. Eagerly this was ~N²/2 (every
        # term was live from bond 1), which is exactly what made the sweep Θ(N·M).
        @test sum(counts) <= 4 * N
        return counts
    end
    @test allequal(maximum.(livecounts))       # the peak is independent of the chain length
end
