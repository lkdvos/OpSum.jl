using Test
using OpSum
using OpSum: TermTable, factor_at, arity, nterms, ispassthrough,
    Term, Terms, scalarop, couple, mpo, instantiate, opsum
using OpSum: Letter
using TensorKit
using LinearAlgebra: dot

include(joinpath(@__DIR__, "..", "testutils.jl"))   # LO

su2 = SU2Space(1 // 2 => 1)
u1 = Rep[U₁](0 => 1, 1 => 1)

@testset "TermTable — flat term storage" begin

    @testset "construction sanity" begin
        H = Terms((dot(spinvector(unitbasis(su2))[i], spinvector(unitbasis(su2))[i + 1]) for i in 1:3))
        tt = TermTable(H)
        @test (nterms(tt), arity(tt)) == (length(H), 2)
        for t in 1:nterms(tt)
            occ = filter(!=(0), tt.sites[:, t])
            @test issorted(occ) && allunique(occ)
        end
    end

    @testset "factor_at reconstructs the running-bond pass-through path" begin
        tt = TermTable(Terms(dot(spinvector(unitbasis(su2))[1], spinvector(unitbasis(su2))[3])))
        @test nterms(tt) == 1
        path = [factor_at(tt, 1, s) for s in 1:4]

        # active sites carry the letter and the outgoing (running) bond charge
        @test !ispassthrough(path[1].letter)
        @test path[1].letter.c == SU2Irrep(1)
        @test path[1].bond == SU2Irrep(1)          # bond out of site 1 (the coupled pair line)
        @test !ispassthrough(path[3].letter)
        @test path[3].bond == SU2Irrep(0)          # total (singlet) out of site 3

        # idle sites are pass-through, carrying the running bond charge to their left
        @test ispassthrough(path[2].letter) && path[2].bond == SU2Irrep(1)
        @test ispassthrough(path[4].letter) && path[4].bond == SU2Irrep(0)
    end

    @testset "U(1) charge threads through idle sites" begin
        raise = LO(Letter(U1Irrep(1), 1))
        lower = LO(Letter(U1Irrep(-1), 1))
        tt = TermTable(Terms(dot(raise[1], lower[4])))
        path = [factor_at(tt, 1, s) for s in 1:5]
        @test all(s -> path[s].bond == U1Irrep(1), 2:3)
        @test path[4].bond == U1Irrep(0)
        @test path[5].bond == U1Irrep(0)
    end

end

# `Terms` is in normal form at all times: coincident terms are summed as they are combined.
@testset "coincident terms are summed" begin
    S = spinvector(unitbasis(su2))
    t = dot(S[1], S[2])

    H = Terms(t, t)
    @test length(H) == 1                      # two coincident terms are one term
    @test only(H).coeff ≈ 2 * only(t).coeff
    @test nterms(TermTable(H)) == 1

    # every combining route sums coincident terms, and `append!` keeps the normal form in place
    @test length(t + t) == 1 && only(t + t).coeff ≈ 2 * only(t).coeff
    @test length(2 * t - t) == 1 && only(2 * t - t).coeff ≈ only(t).coeff
    G = Terms(only(t))
    @test append!(G, t) === G && length(G) == 1 && only(G).coeff ≈ 2 * only(t).coeff
    @test length(append!(G, -2 * t)) == 0

    @test isempty(Terms(t, -t))               # exact cancellation drops the term
    @test length(Terms(t, 2 * t, -3 * t)) == 0
    @test_throws ArgumentError instantiate(opsum(fill(unitbasis(su2), 2), Terms(t, -t)))

    # every accumulation route agrees, whatever its cost
    terms = [dot(S[i], S[i + 1]) for i in 1:5]
    ref = Terms(terms)
    @test length(ref) == 5
    @test ref ≈ Terms(dot(S[i], S[i + 1]) for i in 1:5)
    @test ref ≈ foldl(+, terms; init = Terms{SU2Irrep}())
    @test ref ≈ append!(Terms{SU2Irrep}(), terms)
    @test ref ≈ Terms(only(terms[1]), terms[2:5])
    # nested iterables flatten
    @test ref ≈ Terms([terms[1:2], terms[3:5]])
    @test ref ≈ Terms(terms[1:2], terms[3:5])

    # `+` copies, so older values are never disturbed by a later addition
    a = Terms(terms[1:2])
    b = a + terms[3]
    c = a + terms[4]
    @test length(a) == 2 && length(b) == 3 && length(c) == 3
    @test b ≈ Terms(terms[1:3])
    @test c ≈ Terms(terms[[1, 2, 4]])
    @test a ≈ Terms(terms[1:2])

    # `zero` / `empty` are fresh empty bags; `one` is the no-site term, at instance and type level
    @test isempty(zero(a)) && isempty(empty(a))
    @test ref ≈ append!(zero(ref), terms)
    @test length(one(a)) == 1 && isempty(only(one(a)).sites)

    # mixed arity: the table reports the true maximum
    mixed = Terms(
        dot(S[1], S[2]),
        couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3])
    )
    @test arity(TermTable(mixed)) == 3
    @test length(mixed) == 2
end

@testset "the lattice enters when terms meet an OperatorSum" begin
    S = spinvector(unitbasis(su2))
    terms = [dot(S[i], S[i + 1]) for i in 1:3]
    sites = fill(unitbasis(su2), 4)

    H = Terms(terms)
    @test mpo(opsum(sites, H)).bondsectors == mpo(opsum(FiniteChain(unitbasis(su2), 4), H)).bondsectors

    @test_throws ArgumentError mpo(opsum(fill(unitbasis(su2), 3), H))                 # term past the lattice
    @test_throws ArgumentError mpo(opsum(fill(unitbasis(u1), 4), H))                  # wrong sector type
    @test_throws ArgumentError mpo(opsum(fill(unitbasis(SU2Space(0 => 1)), 4), H))    # no spin-1 letter there

    H2 = append!(Terms(terms), dot(S[1], S[5]))
    @test length(H2) == 4
    @test_throws ArgumentError mpo(opsum(sites, H2))

    # `OperatorSum` takes a lattice or a `Vector{<:Basis}`
    @test mpo(opsum!(OperatorSum(sites), H)).bondsectors == mpo(opsum(sites, H)).bondsectors
    @test_throws MethodError OperatorSum([1, 2, 3, 4])
end

@testset "container and display surface" begin
    S = spinvector(unitbasis(su2))
    t = dot(S[1], S[2])
    u = dot(S[2], S[3])
    H = Terms(t, 2 * u)

    @test length(H) == 2
    @test collect(H) == H.terms                # iteration is the canonical list
    @test eltype(H) == Term{SU2Irrep}
    @test arity(H[1]) == 2 && arity(H[end]) == 2
    @test H[1] == only(t)                      # sorted by sites, so (1,2) comes first
    @test H[1].coeff ≈ only(t).coeff

    # `==` is exact where `≈` is approximate, and both take the normal form first
    @test H == Terms(t, 2 * u)
    @test H == (H + t) - t                     # appended-then-cancelled is the same value
    @test H != Terms(t)
    @test Terms(1.0e-14 * t) != Terms(0.0 * t)
    @test Terms(1.0e-14 * t) ≈ Terms(1.0e-14 * t)

    # both multiplication orders, and division
    @test H * 2 == 2 * H
    @test (H * 2) / 2 == H
    @test Terms(one(t)) * 3 ≈ Terms(scalarop(3.0, unitbasis(su2))[1])

    # a `Term` is comparable and hashable on its content, the coefficient included
    @test only(t) == only(1 * t)
    @test hash(only(t)) == hash(only(1 * t))
    @test only(t) != only(2 * t)
    @test OpSum.support(only(t)) == OpSum.support(only(2 * t))
    @test only(t) != only(u)

    str = sprint(show, H)
    @test startswith(str, "Terms(")
    @test occursin("sites=[1, 2]", str)
end

# Building the MPO and reading a `Terms` never change what the user holds.
@testset "mpo(H) and observers do not mutate" begin
    S = spinvector(unitbasis(su2))
    terms = [dot(S[i], S[i + 1]) for i in 1:4]
    sites = fill(unitbasis(su2), 5)
    H = opsum(sites, reverse(terms), terms[2])   # out of site order, one term coincident
    @test issorted(H.terms.terms) && length(H) == 4
    before = copy(H)
    order = copy(H.terms.terms)
    for alg in (nothing, IndependentSVD(), SequentialSVD())
        alg === nothing ? mpo(H) : mpo(H, alg)
        @test H.terms == before.terms
        @test H.terms.terms == order            # same terms, same order
    end

    h = H.terms
    v = copy(h.terms)
    length(h); collect(h); h[1]; h == copy(h); h ≈ h; sprint(show, h)
    @test h.terms == v
    @test OpSum.dense(h, sites) ≈ OpSum.dense(copy(h), sites)
end
