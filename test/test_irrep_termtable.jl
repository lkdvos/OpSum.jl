using Test
using OpSum
using OpSum: ITOTermTable, _op_at_ito, arity, nterms, nvertices, ispassthrough,
    Term, Terms, TermSum, spin, scalarop, couple, nterms_raw, irrep_mpo, instantiate,
    canonicalize!, opsum, lattice
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using LinearAlgebra: dot

include(joinpath(@__DIR__, "testutils.jl"))   # LO

su2 = SU2Space(1 // 2 => 1)
u1 = Rep[U₁](0 => 1, 1 => 1)

@testset "ITOTermTable — flat ITO storage" begin

    @testset "construction sanity" begin
        H = opsum(fill(su2, 4), (dot(spin(su2)[i], spin(su2)[i + 1]) for i in 1:3))
        tt = ITOTermTable(H)
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
        tt = ITOTermTable(opsum(fill(su2, 4), dot(spin(su2)[1], spin(su2)[3])))
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
        tt = ITOTermTable(opsum(fill(su2, 3), scalarop(2.0, su2)[1]))
        @test nterms(tt) == 1
        @test all(==(0), tt.sites)                 # no active factors
        path = [_op_at_ito(tt, 1, s) for s in 1:3]
        @test all(k -> ispassthrough(k.op) && k.bond == unit(SU2Irrep), path)
    end

    @testset "U(1) charge threads through idle sites" begin
        raise = LO(IrrepOperator(U1Irrep(1), 1))
        lower = LO(IrrepOperator(U1Irrep(-1), 1))
        # sites 1,4 on a 5-site chain
        tt = ITOTermTable(opsum(fill(u1, 5), dot(raise[1], lower[4])))
        path = [_op_at_ito(tt, 1, s) for s in 1:5]
        # after the +1 at site 1, the running bond charge is +1 until the -1 at site 4 closes it
        @test all(s -> path[s].bond == U1Irrep(1), 2:3)
        @test path[4].bond == U1Irrep(0)
        @test path[5].bond == U1Irrep(0)
    end

end

# A `TermSum` is a bag until `canonicalize!` sorts it and merges coincident terms. Nothing outside may
# observe the pre-merge append count.
@testset "append-then-canonicalise" begin
    S = spin(su2)
    sites2 = fill(su2, 2)
    t = dot(S[1], S[2])

    H = opsum(sites2, t, t)
    @test nterms_raw(H) == 2                  # two appended terms …
    @test length(H) == 1                      # … but one term
    @test only(H).coeff ≈ 2 * only(t).coeff
    @test nterms(ITOTermTable(H)) == 1

    # canonicalisation is in place and idempotent
    @test nterms_raw(canonicalize!(H)) == 1
    @test length(canonicalize!(H)) == 1

    @test isempty(opsum(sites2, t, -t))       # exact cancellation drops the term
    @test length(opsum(sites2, t, 2 * t, -3 * t)) == 0
    @test_throws ArgumentError instantiate(opsum(sites2, t, -t))

    # every accumulation route agrees, whatever its cost
    sites6 = fill(su2, 6)
    terms = [dot(S[i], S[i + 1]) for i in 1:5]
    ref = opsum(sites6, terms)
    @test length(ref) == 5
    @test ref ≈ opsum(sites6, (dot(S[i], S[i + 1]) for i in 1:5))
    @test ref ≈ foldl(+, terms; init = opsum(sites6))
    @test ref ≈ append!(opsum(sites6), terms)
    # nested iterables flatten
    @test ref ≈ opsum(sites6, [terms[1:2], terms[3:5]])
    @test ref ≈ opsum(sites6, terms[1:2], terms[3:5])

    # `+` copies, so older values are never disturbed by a later addition
    a = opsum(sites6, terms[1:2])
    b = a + terms[3]
    c = a + terms[4]
    @test length(a) == 2 && length(b) == 3 && length(c) == 3
    @test b ≈ opsum(sites6, terms[1:3])
    @test c ≈ opsum(sites6, terms[[1, 2, 4]])
    @test a ≈ opsum(sites6, terms[1:2])

    # mixed arity: the table reports the true maximum
    mixed = opsum(
        fill(su2, 3),
        dot(S[1], S[2]),
        couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3])
    )
    @test arity(ITOTermTable(mixed)) == 3
    @test length(mixed) == 2
end

@testset "the lattice travels with the operator" begin
    S = spin(su2)
    terms = [dot(S[i], S[i + 1]) for i in 1:3]
    sites = fill(su2, 4)

    H = opsum(sites, terms)
    @test lattice(H) == sites
    @test length(lattice(H)) == nvertices(ITOTermTable(H))
    @test irrep_mpo(H)[2] == irrep_mpo(opsum(sites, terms))[2]

    # the checks binding exists for — the only place letters and spaces are confronted
    @test_throws ArgumentError opsum(fill(su2, 3), terms)                   # term past the lattice
    @test_throws ArgumentError opsum(fill(u1, 4), terms)                    # wrong sector type
    @test_throws ArgumentError opsum(fill(SU2Space(0 => 1), 4), terms)      # no spin-1 letter there
    @test_throws ArgumentError H + dot(S[1], S[5])                          # checked on the way in
    @test_throws ArgumentError append!(opsum(sites), dot(S[1], S[5]))

    # adding operators on different lattices is refused
    @test_throws ArgumentError H + opsum(fill(su2, 5), dot(S[1], S[2]))

    # `sites` need not already be a `Vector{<:ElementarySpace}`
    @test lattice(opsum((su2 for _ in 1:4), terms)) == sites
    @test lattice(opsum(ntuple(_ -> su2, 4), terms)) == sites
    @test_throws ArgumentError opsum([1, 2, 3, 4], terms)

    # the empty operator on a lattice
    @test isempty(opsum(sites))
    @test lattice(opsum(sites)) == sites
end

# The container surface a consumer actually touches. All of it goes through the normal form.
@testset "container and display surface" begin
    S = spin(su2)
    sites = fill(su2, 3)
    t = dot(S[1], S[2])
    u = dot(S[2], S[3])
    H = opsum(sites, t, 2 * u)

    @test length(H) == 2
    @test collect(H) == H.terms                # iteration is the canonical list
    @test eltype(H) == Term{SU2Irrep}
    @test arity(H[1]) == 2 && arity(H[end]) == 2
    @test H[1] == only(t)                      # sorted by sites, so (1,2) comes first
    @test H[1].coeff ≈ only(t).coeff

    # `==` is exact where `≈` is approximate, and both take the normal form first
    @test H == opsum(sites, t, 2 * u)
    @test H == (H + t) - t                     # appended-then-cancelled is the same value
    @test H != opsum(sites, t)
    @test opsum(sites, 1.0e-14 * t) != opsum(sites, 0.0 * t)
    @test opsum(sites, 1.0e-14 * t) ≈ opsum(sites, 1.0e-14 * t)

    # both multiplication orders, and division
    @test H * 2 == 2 * H
    @test (H * 2) / 2 == H
    @test -H == -1 * H
    @test opsum(sites, one(t)) * 3 ≈ opsum(sites, scalarop(3.0, su2)[1])

    # a `Term` is comparable and hashable on its content, ignoring the coefficient
    @test only(t) == only(2 * t)
    @test hash(only(t)) == hash(only(2 * t))
    @test only(t) != only(u)
    @test length(Set([only(t), only(2 * t), only(u)])) == 2

    str = sprint(show, H)
    @test startswith(str, "TermSum(")
    @test occursin("sites=[1, 2]", str)
    @test occursin(" + ", str)                 # two terms, joined
    @test sprint(show, opsum(sites)) == "TermSum()"
    @test startswith(sprint(show, t), "Terms(")
    @test startswith(sprint(show, only(t)), "Term(sites=")
end
