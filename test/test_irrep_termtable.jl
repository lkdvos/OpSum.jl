using Test
using OpSum
using OpSum: ITOTermTable, _op_at_ito, arity, nterms, nvertices, ispassthrough,
    Term, Terms, spin, scalarop, couple, nterms_raw, irrep_mpo, instantiate,
    canonicalize!, opsum
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using LinearAlgebra: dot

include(joinpath(@__DIR__, "testutils.jl"))   # LO

su2 = SU2Space(1 // 2 => 1)
u1 = Rep[U₁](0 => 1, 1 => 1)

@testset "ITOTermTable — flat ITO storage" begin

    @testset "construction sanity" begin
        H = opsum((dot(spin(su2)[i], spin(su2)[i + 1]) for i in 1:3))
        tt = ITOTermTable(H, 4)
        @test nvertices(tt) == 4
        @test nterms(tt) == length(H)
        @test arity(tt) == 2                      # two-body terms
        for t in 1:nterms(tt)
            occ = filter(!=(0), tt.sites[:, t])
            @test issorted(occ) && allunique(occ)
        end
    end

    @testset "_op_at_ito reconstructs the running-bond pass-through path" begin
        # a single two-body term on sites (1,3) of a 4-site chain, coupled to a singlet
        tt = ITOTermTable(opsum(dot(spin(su2)[1], spin(su2)[3])), 4)
        @test nterms(tt) == 1
        path = [_op_at_ito(tt, 1, s) for s in 1:4]

        # active sites carry the ITO letter and the outgoing (running) bond charge
        @test !ispassthrough(path[1].op)
        @test path[1].op.c == SU2Irrep(1)
        @test path[1].bond == SU2Irrep(1)          # bond out of site 1 (the coupled pair line)
        @test !ispassthrough(path[3].op)
        @test path[3].bond == SU2Irrep(0)          # total (singlet) out of site 3

        # idle sites are pass-through, carrying the running bond charge to their left
        @test ispassthrough(path[2].op) && path[2].bond == SU2Irrep(1)
        @test ispassthrough(path[4].op) && path[4].bond == SU2Irrep(0)
    end

    @testset "K=0 identity term is all pass-through (trivial running charge)" begin
        tt = ITOTermTable(opsum(scalarop(2.0, su2)[1]), 3)
        @test nterms(tt) == 1
        @test all(==(0), tt.sites)                 # no active factors
        path = [_op_at_ito(tt, 1, s) for s in 1:3]
        @test all(k -> ispassthrough(k.op) && k.bond == unit(SU2Irrep), path)
    end

    @testset "U(1) charge threads through idle sites" begin
        raise = LO(IrrepOperator(U1Irrep(1), 1))
        lower = LO(IrrepOperator(U1Irrep(-1), 1))
        # sites 1,4 on a 5-site chain
        tt = ITOTermTable(opsum(dot(raise[1], lower[4])), 5)
        path = [_op_at_ito(tt, 1, s) for s in 1:5]
        # after the +1 at site 1, the running bond charge is +1 until the -1 at site 4 closes it
        @test all(s -> path[s].bond == U1Irrep(1), 2:3)
        @test path[4].bond == U1Irrep(0)
        @test path[5].bond == U1Irrep(0)
    end

end

# A bag until `canonicalize!` merges it; the pre-merge append count must not be observable.
@testset "append-then-canonicalise" begin
    S = spin(su2)
    sites2 = fill(su2, 2)
    t = dot(S[1], S[2])

    H = opsum(t, t)
    @test nterms_raw(H) == 2                  # two appended terms …
    @test length(H) == 1                      # … but one term
    @test only(H).coeff ≈ 2 * only(t).coeff
    @test nterms(ITOTermTable(H, 2)) == 1

    # canonicalisation is in place and idempotent
    @test nterms_raw(canonicalize!(H)) == 1
    @test length(canonicalize!(H)) == 1

    @test isempty(opsum(t, -t))               # exact cancellation drops the term
    @test length(opsum(t, 2 * t, -3 * t)) == 0
    @test_throws ArgumentError instantiate(opsum(t, -t), sites2)

    # every accumulation route agrees, whatever its cost
    sites6 = fill(su2, 6)
    terms = [dot(S[i], S[i + 1]) for i in 1:5]
    ref = opsum(terms)
    @test length(ref) == 5
    @test ref ≈ opsum(dot(S[i], S[i + 1]) for i in 1:5)
    @test ref ≈ foldl(+, terms; init = Terms{SU2Irrep}())
    @test ref ≈ append!(Terms{SU2Irrep}(), terms)
    # a bare `Term` is accepted too, including as the argument that fixes the sector type
    @test ref ≈ opsum(only(terms[1]), terms[2:5])
    @test onlyterm(opsum(only(terms[1]))) == only(terms[1])
    # nested iterables flatten
    @test ref ≈ opsum([terms[1:2], terms[3:5]])
    @test ref ≈ opsum(terms[1:2], terms[3:5])

    # `+` copies, so older values are never disturbed by a later addition
    a = opsum(terms[1:2])
    b = a + terms[3]
    c = a + terms[4]
    @test length(a) == 2 && length(b) == 3 && length(c) == 3
    @test b ≈ opsum(terms[1:3])
    @test c ≈ opsum(terms[[1, 2, 4]])
    @test a ≈ opsum(terms[1:2])

    # the non-mutating container protocol: `copy` owns its vector, so the mutating routes
    # (`append!`, `canonicalize!`) cannot reach back into the original
    d = append!(copy(a), terms[3])
    @test length(a) == 2 && length(d) == 3
    @test a ≈ opsum(terms[1:2])
    @test d ≈ opsum(terms[1:3])
    @test copy(a) ≈ a
    @test copy(a).terms !== a.terms

    # `zero` / `empty` are fresh empty bags; `one` is the no-site term, at instance and type level
    @test isempty(zero(a)) && isempty(empty(a)) && isempty(zero(Terms{SU2Irrep}))
    @test ref ≈ append!(zero(ref), terms)
    @test zero(a) !== zero(a)
    @test one(Terms{SU2Irrep}) ≈ one(a)
    @test length(one(a)) == 1 && isempty(only(one(a)).sites)
    @test ref + zero(ref) ≈ ref

    # mixed arity: the table reports the true maximum
    mixed = opsum(
        dot(S[1], S[2]),
        couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3])
    )
    @test arity(ITOTermTable(mixed, 3)) == 3
    @test length(mixed) == 2
end

@testset "the lattice enters where the MPO is formed" begin
    S = spin(su2)
    terms = [dot(S[i], S[i + 1]) for i in 1:3]
    sites = fill(su2, 4)

    H = opsum(terms)
    # a bag is latticeless, so nothing about it knows `sites`; the sweep needs only the site count
    @test nvertices(ITOTermTable(H, length(sites))) == length(sites)
    @test irrep_mpo(H, sites).bondsectors == irrep_mpo(H, FiniteChain(su2, 4)).bondsectors

    # the checks the lattice boundary exists for — the only place letters and spaces are confronted
    @test_throws ArgumentError irrep_mpo(H, fill(su2, 3))                 # term past the lattice
    @test_throws ArgumentError irrep_mpo(H, fill(u1, 4))                  # wrong sector type
    @test_throws ArgumentError irrep_mpo(H, fill(SU2Space(0 => 1), 4))    # no spin-1 letter there
    @test_throws ArgumentError instantiate(H, fill(su2, 3))               # and again at instantiate
    @test_throws ArgumentError islossless(H, fill(su2, 3))

    # appending is latticeless, so a term past the eventual lattice is caught at compression
    H2 = append!(opsum(terms), dot(S[1], S[5]))
    @test nterms_raw(H2) == 4
    @test_throws ArgumentError irrep_mpo(H2, sites)

    # `lat` need not already be a `FiniteChain` or a `Vector{<:ElementarySpace}`
    @test irrep_mpo(H, (su2 for _ in 1:4)).bondsectors == irrep_mpo(H, sites).bondsectors
    @test irrep_mpo(H, ntuple(_ -> su2, 4)).bondsectors == irrep_mpo(H, sites).bondsectors
    @test_throws ArgumentError irrep_mpo(H, [1, 2, 3, 4])

    # FiniteChain has an edge where InfiniteChain wraps
    @test length(FiniteChain(su2, 4)) == 4
    @test FiniteChain(sites) == FiniteChain(su2, 4)
    @test_throws BoundsError FiniteChain(su2, 4)[5]
    @test InfiniteChain([su2])[5] == su2

    # the empty operator
    @test isempty(Terms{SU2Irrep}())
    @test_throws ArgumentError opsum()
end

# The container surface a consumer touches; all of it goes through the normal form.
@testset "container and display surface" begin
    S = spin(su2)
    sites = fill(su2, 3)
    t = dot(S[1], S[2])
    u = dot(S[2], S[3])
    H = opsum(t, 2 * u)

    @test length(H) == 2
    @test collect(H) == H.terms                # iteration is the canonical list
    @test eltype(H) == Term{SU2Irrep}
    @test arity(H[1]) == 2 && arity(H[end]) == 2
    @test H[1] == only(t)                      # sorted by sites, so (1,2) comes first
    @test H[1].coeff ≈ only(t).coeff

    # `==` is exact where `≈` is approximate, and both take the normal form first
    @test H == opsum(t, 2 * u)
    @test H == (H + t) - t                     # appended-then-cancelled is the same value
    @test H != opsum(t)
    @test opsum(1.0e-14 * t) != opsum(0.0 * t)
    @test opsum(1.0e-14 * t) ≈ opsum(1.0e-14 * t)

    # both multiplication orders, and division
    @test H * 2 == 2 * H
    @test (H * 2) / 2 == H
    @test -H == -1 * H
    @test opsum(one(t)) * 3 ≈ opsum(scalarop(3.0, su2)[1])

    # a `Term` is comparable and hashable on its content, ignoring the coefficient
    @test only(t) == only(2 * t)
    @test hash(only(t)) == hash(only(2 * t))
    @test only(t) != only(u)
    @test length(Set([only(t), only(2 * t), only(u)])) == 2

    str = sprint(show, H)
    @test startswith(str, "Terms(")
    @test occursin("sites=[1, 2]", str)
    @test occursin(" + ", str)                 # two terms, joined
    @test sprint(show, Terms{SU2Irrep}()) == "Terms()"
    @test startswith(sprint(show, only(t)), "Term(sites=")
end
