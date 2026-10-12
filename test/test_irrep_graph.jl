# Tests for the reduced-MPO sweeps (irrepgraph.jl): VertexCover, SequentialSVD and IndependentSVD.
# Checks: `mpo_terms` round-trip, dense agreement with `instantiate`, pinned per-bond sector profiles,
# the research/persistent-graph-mpo.md §2.2 pending-versus-started collision, and the scaling guard.

using Test
using OpSum
using OpSum: mpo, irrep_mpo_tensors, mpo_terms, instantiate, couple, scalarop, opsum
using OpSum: BipartiteAlgorithm, SVDBondAlgorithm, VertexCover, IndependentSVD, SequentialSVD
using OpSum: ITOTermTable, _irrep_sweep, ITOGraph, _at_site!
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using TensorKit: @tensor
using LinearAlgebra: dot, norm

include(joinpath(@__DIR__, "testutils.jl"))   # LO, physmatrix, islossless

# densified operator check, unaffected by the per-bond rotation of the SVD (`mpo_terms` is not reliable there)
contract2(T) = @tensor Op[o1 o2 bL; bR i1 i2] := T[1][bL o1; i1 bm] * T[2][bm o2; i2 bR]
function contract3(T)
    return @tensor Op[o1 o2 o3 bL; bR i1 i2 i3] :=
        T[1][bL o1; i1 b1] * T[2][b1 o2; i2 b2] * T[3][b2 o3; i3 bR]
end
contractN(T) = length(T) == 2 ? contract2(T) : contract3(T)

# Per-bond `[charge => multiplicity]`, sorted and stringified; the order within a bond is not pinned.
function bondprofile(secs)
    return map(secs) do sec
        d = Dict{String, Int}()
        for c in sec
            d[string(c)] = get(d, string(c), 0) + 1
        end
        return sort!([k => v for (k, v) in d])
    end
end

mpo_operator(Ws, secs, sites) = instantiate(opsum(sites, mpo_terms(Ws, secs)))

# Reference Hamiltonians (SU2, U1, trivial; K ∈ 0..3) as `(name, terms, lattice)`.
function reference_hamiltonians()
    cases = Tuple{String, Terms, FiniteChain}[]
    let V = SU2Space(1 // 2 => 1)
        S = spinvector(unitbasis(V))
        for N in (2, 3, 4, 5, 6)
            H = opsum(dot(S[i], S[i + 1]) for i in 1:(N - 1))
            push!(cases, ("SU2 Heisenberg N=$N", H, FiniteChain(unitbasis(V), N)))
        end
        push!(cases, ("SU2 fields", opsum(S[1], S[2], S[3]), FiniteChain(unitbasis(V), 3)))
        push!(
            cases, (
                "SU2 decoupled pairs N=6",
                opsum(dot(S[1], S[2]), dot(S[3], S[4]), dot(S[5], S[6])), FiniteChain(unitbasis(V), 6),
            )
        )
        push!(
            cases, (
                "SU2 K=3 three-body",
                opsum(couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0))),
                FiniteChain(unitbasis(V), 3),
            )
        )
        # coupling S₁⊗S₂ to charge 0 makes the tail merge with the on-site S₄ (see the collision testset)
        push!(
            cases, (
                "SU2 K=3 tail collides with on-site field",
                opsum(
                    couple(couple(S[1], S[2]; to = SU2Irrep(0)), S[4]; to = SU2Irrep(1)),
                    2.0 * S[4],
                ),
                FiniteChain(unitbasis(V), 4),
            )
        )
        # boundary shapes: nothing active at site 1; first active site is the last one
        push!(cases, ("SU2 no term starts at site 1", opsum(dot(S[2], S[3])), FiniteChain(unitbasis(V), 3)))
        push!(cases, ("SU2 first active site == N", opsum(S[1], S[4]), FiniteChain(unitbasis(V), 4)))
    end
    let V = Rep[U₁](0 => 1, 1 => 1)
        raise = LO(IrrepOperator(U1Irrep(1), 1))
        lower = LO(IrrepOperator(U1Irrep(-1), 1))
        for N in (3, 4)
            H = opsum(
                (dot(raise[i], lower[i + 1]) for i in 1:(N - 1)),
                (dot(lower[i], raise[i + 1]) for i in 1:(N - 1)),
            )
            push!(cases, ("U1 hopping N=$N", H, FiniteChain(unitbasis(V), N)))
        end
        push!(
            cases, (
                "U1 K=3 three-body",
                opsum(
                    couple(couple(raise[1], raise[2]; to = U1Irrep(2)), lower[3]; to = U1Irrep(1))
                ),
                FiniteChain(unitbasis(V), 3),
            )
        )
        # a charge-0 letter leaves the running bond trivial: like `n̂` colliding with the tail of `n̂ᵢn̂ⱼ`
        z = LO(IrrepOperator(U1Irrep(0), 2))   # a matrix unit; index 1 is the identity
        push!(
            cases, (
                "U1 charge-0 on-site collides with two-site tail",
                opsum(couple(z[1], z[3]; to = U1Irrep(0)), 0.5 * z[3]),
                FiniteChain(unitbasis(V), 3),
            )
        )
    end
    let V = ℂ^2
        ops = [IrrepOperator(unit(Trivial), n) for n in 1:4]
        H = opsum(couple(LO(ops[2])[i], LO(ops[3])[i + 1]; to = unit(Trivial)) for i in 1:2)
        push!(cases, ("trivial sector N=3", H, FiniteChain(unitbasis(V), 3)))
        push!(
            cases, (
                "trivial on-site collides with two-site tail",
                opsum(
                    couple(LO(ops[2])[1], LO(ops[3])[3]; to = unit(Trivial)),
                    0.5 * LO(ops[3])[3],
                ),
                FiniteChain(unitbasis(V), 3),
            )
        )
    end
    return cases
end

# The compression's pinned output: a lost suffix-class merge inflates a bond here and nowhere else.
const EXPECTED_BONDS = Dict{String, Vector{Vector{Pair{String, Int}}}}(
    "SU2 Heisenberg N=2" => [["Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1]],
    "SU2 Heisenberg N=3" => [["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1]],
    "SU2 Heisenberg N=4" => [["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 2, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1]],
    "SU2 Heisenberg N=5" => [["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 2, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 2, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1]],
    "SU2 Heisenberg N=6" => [["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 2, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 2, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 2, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1]],
    "SU2 fields" => [["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](1)" => 1]],
    "SU2 decoupled pairs N=6" => [["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 2], ["Irrep[SU₂](0)" => 2, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 2], ["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1]],
    "SU2 K=3 three-body" => [["Irrep[SU₂](1)" => 1], ["Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1]],
    "SU2 K=3 tail collides with on-site field" => [["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1], ["Irrep[SU₂](0)" => 1], ["Irrep[SU₂](1)" => 1]],
    "SU2 no term starts at site 1" => [["Irrep[SU₂](0)" => 1], ["Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1]],
    "SU2 first active site == N" => [["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](1)" => 1]],
    "U1 hopping N=3" => [["Irrep[U₁](-1)" => 1, "Irrep[U₁](0)" => 1, "Irrep[U₁](1)" => 1], ["Irrep[U₁](-1)" => 1, "Irrep[U₁](0)" => 1, "Irrep[U₁](1)" => 1], ["Irrep[U₁](0)" => 1]],
    "U1 hopping N=4" => [["Irrep[U₁](-1)" => 1, "Irrep[U₁](0)" => 1, "Irrep[U₁](1)" => 1], ["Irrep[U₁](-1)" => 1, "Irrep[U₁](0)" => 2, "Irrep[U₁](1)" => 1], ["Irrep[U₁](-1)" => 1, "Irrep[U₁](0)" => 1, "Irrep[U₁](1)" => 1], ["Irrep[U₁](0)" => 1]],
    "U1 K=3 three-body" => [["Irrep[U₁](1)" => 1], ["Irrep[U₁](2)" => 1], ["Irrep[U₁](1)" => 1]],
    "U1 charge-0 on-site collides with two-site tail" => [["Irrep[U₁](0)" => 1], ["Irrep[U₁](0)" => 1], ["Irrep[U₁](0)" => 1]],
    "trivial sector N=3" => [["Trivial()" => 2], ["Trivial()" => 2], ["Trivial()" => 1]],
    "trivial on-site collides with two-site tail" => [["Trivial()" => 1], ["Trivial()" => 1], ["Trivial()" => 1]],
    "pure identity" => [["Irrep[SU₂](0)" => 1], ["Irrep[SU₂](0)" => 1], ["Irrep[SU₂](0)" => 1]],
    "identity plus bond" => [["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1], ["Irrep[SU₂](0)" => 1]],
    # `mpo` forces the finish channel into the cover, costing one index at bond 1 (unconstrained cover pinned below)
    "identity plus two bonds" => [["Irrep[SU₂](0)" => 2, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1]],
)

const HEISENBERG8_BONDS = [["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 2, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 2, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 2, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 2, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 2, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1]]

@testset "VertexCover sweep — lossless, dense-exact, pinned bond sectors" begin
    for (name, H, sites) in reference_hamiltonians()
        N = length(sites)
        Ws, secs = reduced(mpo(opsum(sites, H), BipartiteAlgorithm()))
        @testset "$name" begin
            @test mpo_terms(Ws, secs) ≈ H
            @test bondprofile(secs) == EXPECTED_BONDS[name]
            # dense oracle; `physmatrix` needs a dim-1 total charge, `contractN` covers N ∈ {2, 3}
            if 2 <= N <= 3 && all(c -> dim(c) == 1, secs[end])
                d = dim(TensorKit.space(sites[1]))
                Mmpo = physmatrix(contractN(irrep_mpo_tensors(Ws, secs, sites)), N, d)
                @test Mmpo ≈ physmatrix(instantiate(opsum(sites, H)), N, d)
            end
        end
    end
end

@testset "the default selector is BipartiteAlgorithm / VertexCover" begin
    V = SU2Space(1 // 2 => 1)
    N = 4
    sites = fill(unitbasis(V), N)
    H = opsum((dot(spinvector(unitbasis(V))[i], spinvector(unitbasis(V))[i + 1]) for i in 1:(N - 1)))
    Wdef, sdef = reduced(mpo(opsum(sites, H)))
    Wsel, ssel = reduced(mpo(opsum(sites, H), BipartiteAlgorithm()))
    Wstr, sstr = _irrep_sweep(ITOTermTable(H, N), N, VertexCover())
    @test bondprofile(sdef) == bondprofile(ssel) == bondprofile(sstr)
    @test mpo_operator(Wdef, sdef, sites) ≈ mpo_operator(Wsel, ssel, sites)
    @test mpo_operator(Wstr, sstr, sites) ≈ mpo_operator(Wsel, ssel, sites)
    T = irrep_mpo_tensors(Wsel, ssel, sites)
    @test T isa Vector{<:AbstractTensorMap}
    @test length(T) == N
end

@testset "SequentialSVD (lossless) reconstructs the operator" begin
    # The SVD rotates each bond's basis, so compare dense tensors (N ≤ 3), not `mpo_terms`.
    for (name, H, sites) in reference_hamiltonians()
        N = length(sites)
        2 <= N <= 3 || continue
        d = dim(TensorKit.space(sites[1]))
        Wg, sg = reduced(mpo(opsum(sites, H), SVDBondAlgorithm(; sweep = SequentialSVD)))
        # needs a dim-1 total charge (excludes the spin-1 field sum)
        all(c -> dim(c) == 1, sg[end]) || continue
        Mgraph = physmatrix(contractN(irrep_mpo_tensors(Wg, sg, sites)), N, d)
        Mexact = physmatrix(instantiate(opsum(sites, H)), N, d)
        @testset "$name" begin
            @test Mgraph ≈ Mexact
        end
    end
end

@testset "the two lossless SVD sweeps agree on the internal bonds" begin
    # Bond N is excluded: the independent sweep hardcodes it to `unit(I)`, the sequential one reports the total charge.
    for (name, H, sites) in reference_hamiltonians()
        N = length(sites)
        N >= 2 || continue
        _, si = reduced(mpo(opsum(sites, H), SVDBondAlgorithm(; sweep = IndependentSVD)))
        _, ss = reduced(mpo(opsum(sites, H), SVDBondAlgorithm(; sweep = SequentialSVD)))
        @testset "$name" begin
            @test bondprofile(si[1:(N - 1)]) == bondprofile(ss[1:(N - 1)])
        end
    end
end

@testset "suffix-merge over many site steps" begin
    V = SU2Space(1 // 2 => 1)
    N = 8
    sites = fill(unitbasis(V), N)
    H = opsum((dot(spinvector(unitbasis(V))[i], spinvector(unitbasis(V))[i + 1]) for i in 1:(N - 1)))
    Ws, secs = reduced(mpo(opsum(sites, H), BipartiteAlgorithm()))
    @test bondprofile(secs) == HEISENBERG8_BONDS
    @test islossless(opsum(sites, H))
end

@testset "K=0 identity terms" begin
    V = SU2Space(1 // 2 => 1)
    S = spinvector(unitbasis(V))
    sites = fill(unitbasis(V), 3)
    for (name, H) in (
            ("pure identity", opsum(scalarop(2.5, unitbasis(V))[1])),
            ("identity plus bond", opsum(scalarop(2.5, unitbasis(V))[1], dot(S[1], S[2]))),
            (
                "identity plus two bonds",
                opsum(scalarop(-1.5, unitbasis(V))[1], dot(S[1], S[2]), dot(S[2], S[3])),
            ),
        )
        Ws, secs = reduced(mpo(opsum(sites, H), BipartiteAlgorithm()))
        @testset "$name" begin
            @test bondprofile(secs) == EXPECTED_BONDS[name]
            @test islossless(opsum(sites, H))
            @test mpo_tensormap(irrep_mpo_tensors(Ws, secs, sites)) ≈ instantiate(opsum(sites, H))
        end
    end

    # without the forced finish channel the cover is one index smaller at bond 1
    H = opsum(FiniteChain(sites), scalarop(-1.5, unitbasis(V))[1], dot(S[1], S[2]), dot(S[2], S[3]))
    tt = OpSum.ITOTermTable(H.terms, 3)
    Ws, secs = OpSum._irrep_sweep(tt, 3, OpSum.VertexCover())
    @test bondprofile(secs) == [["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1, "Irrep[SU₂](1)" => 1], ["Irrep[SU₂](0)" => 1]]
    @test mpo_terms(Ws, secs) ≈ H.terms
end

@testset "suffix signature needs the running bond charge" begin
    # A suffix class is `(remaining factors, running bond charge)`; dropping the charge merges terms that
    # share the spin-1 letter at site 4 but reach bond 2 at charge 0 and 1, and trips the sector-purity check.
    # The site has a non-identity charge-0 letter (|½⟩⟨½|) next to the charge-1 one.
    V = SU2Space(0 => 1, 1 // 2 => 1)
    zl, sl = IrrepOperator(SU2Irrep(0), 2), IrrepOperator(SU2Irrep(1), 1)   # charge-0 and charge-1 letters
    @test zl.c == SU2Irrep(0) && sl.c == SU2Irrep(1)
    z, s = LO(zl), LO(sl)
    N = 4
    sites = fill(unitbasis(V), N)
    H = opsum(
        couple(z[1], s[4]; to = SU2Irrep(1)),
        2.0 * couple(s[1], s[4]; to = SU2Irrep(1)),
    )
    tt = ITOTermTable(H, N)

    # the two classes agree on the remaining factors at bond 2 but not on the running charge
    @test OpSum._op_at_ito(tt, 1, 4) == OpSum._op_at_ito(tt, 2, 4)
    @test OpSum._op_at_ito(tt, 1, 3) != OpSum._op_at_ito(tt, 2, 3)

    Wg, sg = reduced(mpo(opsum(sites, H), BipartiteAlgorithm()))
    back = mpo_terms(Wg, sg)
    @test back ≈ H
    @test norm(instantiate(opsum(sites, back)) - instantiate(opsum(sites, H))) < 1.0e-10
end

@testset "pending suffix class can collide with a started one" begin
    # `A₁B₃` and the on-site `B₃` share a suffix class from site 2 on, so every bond stays 1-dimensional;
    # naive lazy insertion gives [2, 2, 1]. No showcase model triggers this.
    V = ℂ^2
    ops = [IrrepOperator(unit(Trivial), n) for n in 1:4]
    A, B = LO(ops[2]), LO(ops[3])
    N = 3
    sites = fill(unitbasis(V), N)
    H = opsum(couple(A[1], B[3]; to = unit(Trivial)), 0.5 * B[3])
    tt = ITOTermTable(H, N)

    @test OpSum._op_at_ito(tt, 1, 2) == OpSum._op_at_ito(tt, 2, 2)
    @test OpSum._op_at_ito(tt, 1, 3) == OpSum._op_at_ito(tt, 2, 3)
    @test OpSum._op_at_ito(tt, 1, 1) != OpSum._op_at_ito(tt, 2, 1)

    Wg, sg = reduced(mpo(opsum(sites, H), BipartiteAlgorithm()))
    @test length.(sg) == [1, 1, 1]
    @test mpo_terms(Wg, sg) ≈ H
    d = dim(V)
    @test physmatrix(contract3(irrep_mpo_tensors(Wg, sg, sites)), N, d) ≈
        physmatrix(instantiate(opsum(sites, H)), N, d)
end

@testset "live right vertices stay bounded independently of N" begin
    # for a finite-range model the live count is bounded by a constant, not by the number of terms
    V = SU2Space(1 // 2 => 1)
    sizes = (5, 20, 80)
    livecounts = map(sizes) do N
        H = opsum((dot(spinvector(unitbasis(V))[i], spinvector(unitbasis(V))[i + 1]) for i in 1:(N - 1)))
        g = ITOGraph(ITOTermTable(H, N), N)
        counts = Int[]
        for i in 1:N
            _at_site!(g, i, VertexCover())
            push!(counts, length(g.rrepr))
        end
        @test counts[end] == 1                 # only the exhausted ("done") suffix class survives
        # right-vertex visits summed over the sweep are linear in N
        @test sum(counts) <= 4 * N
        return counts
    end
    @test allequal(maximum.(livecounts))       # the peak is independent of the chain length
end

@testset "_invariant is a real throw, not a strippable assert" begin
    # `@assert` would be stripped under `--check-bounds=no`; the helper must keep throwing.
    err = try
        OpSum._invariant("the widget is not sector-pure")
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    msg = sprint(showerror, err)
    @test occursin("OpSum internal invariant violated", msg)
    @test occursin("the widget is not sector-pure", msg)
    @test occursin("bug in OpSum", msg)
end
