using Test
using OpSum
using OpSum: ITOTermTable, _op_at_ito, arity, nterms, nvertices, ispassthrough,
    TermSum, spin, scalarop, couple, nrows, irrep_mpo, instantiate
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using LinearAlgebra: dot

include(joinpath(@__DIR__, "testutils.jl"))   # LO

su2 = SU2Space(1 // 2 => 1)
u1 = Rep[U₁](0 => 1, 1 => 1)

@testset "ITOTermTable — flat ITO storage" begin

    @testset "construction sanity" begin
        H = reduce(+, dot(spin(su2)[i], spin(su2)[i + 1]) for i in 1:3)
        tt = ITOTermTable(H, fill(su2, 4))
        @test nvertices(tt) == 4
        @test nterms(tt) == length(H.terms)
        @test arity(tt) == 2                      # two-body terms
        for t in 1:nterms(tt)
            occ = filter(!=(0), tt.sites[:, t])
            @test issorted(occ) && allunique(occ)
        end
    end

    @testset "_op_at_ito reconstructs the running-bond pass-through path" begin
        # a single two-body term on sites (1,3) of a 4-site chain, coupled to a singlet
        H = dot(spin(su2)[1], spin(su2)[3])
        tt = ITOTermTable(H, fill(su2, 4))
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
        H = scalarop(2.0, su2)[1]
        tt = ITOTermTable(H, fill(su2, 3))
        @test nterms(tt) == 1
        @test all(==(0), tt.sites)                 # no active factors
        path = [_op_at_ito(tt, 1, s) for s in 1:3]
        @test all(k -> ispassthrough(k.op) && k.bond == unit(SU2Irrep), path)
    end

    @testset "U(1) charge threads through idle sites" begin
        raise = LO(IrrepOperator(U1Irrep(1), 1))
        lower = LO(IrrepOperator(U1Irrep(-1), 1))
        H = dot(raise[1], lower[4])                # sites 1,4 on a 5-site chain
        tt = ITOTermTable(H, fill(u1, 5))
        path = [_op_at_ito(tt, 1, s) for s in 1:5]
        # after the +1 at site 1, the running bond charge is +1 until the -1 at site 4 closes it
        @test all(s -> path[s].bond == U1Irrep(1), 2:3)
        @test path[4].bond == U1Irrep(0)
        @test path[5].bond == U1Irrep(0)
    end

end

# The store is append-only: `+` concatenates and duplicates/cancellations are resolved once, lazily.
# Nothing outside may observe the pre-dedup row count.
@testset "append-only accumulation" begin
    S = spin(su2)
    t = dot(S[1], S[2])

    H = t + t
    @test nrows(H) == 2                       # two appended rows …
    @test length(H) == 1                      # … but one term
    @test only(values(H)) ≈ 2 * only(values(t))
    @test nterms(ITOTermTable(H, fill(su2, 2))) == 1

    @test isempty(t - t)                      # exact cancellation drops the term
    @test length(t + 2 * t - 3 * t) == 0
    @test_throws ArgumentError instantiate(t - t, fill(su2, 2))

    terms = [dot(S[i], S[i + 1]) for i in 1:5]
    @test sum(terms) ≈ reduce(+, terms)
    @test sum(terms) ≈ sum(dot(S[i], S[i + 1]) for i in 1:5)
    @test length(sum(terms)) == 5

    # Several term lists share one append buffer. An older value must never see a later append —
    # this is the one invariant the in-place `+` fast path rests on.
    a = sum(terms[1:2])
    b = a + terms[3]
    c = a + terms[4]
    @test length(a) == 2 && length(b) == 3 && length(c) == 3
    @test b ≈ sum(terms[1:3])
    @test c ≈ sum(terms[[1, 2, 4]])
    @test a ≈ sum(terms[1:2])

    # mixed arity: the buffer stride widens, and the table reports the true maximum
    mixed = dot(S[1], S[2]) + couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3])
    @test arity(ITOTermTable(mixed, fill(su2, 3))) == 3
    @test length(mixed) == 2
end

@testset "lattice binding" begin
    S = spin(su2)
    H = sum([dot(S[i], S[i + 1]) for i in 1:3])
    sites = fill(su2, 4)

    @test_throws ArgumentError lattice(H)     # unbound
    @test_throws ArgumentError irrep_mpo(H)
    Hl = onlattice(H, sites)
    @test lattice(Hl) == sites
    @test onlattice(Hl, sites) === Hl                       # idempotent
    @test_throws ArgumentError onlattice(Hl, fill(su2, 5))   # never silently rebind
    @test irrep_mpo(Hl)[2] == irrep_mpo(H, sites)[2]

    # the checks the binding exists for
    @test_throws ArgumentError onlattice(H, fill(su2, 3))    # a term reaches past the lattice
    @test_throws ArgumentError onlattice(H, fill(u1, 4))     # wrong sector type
    @test_throws ArgumentError onlattice(H, fill(SU2Space(0 => 1), 4))  # no spin-1 letter there

    # adding a bound and an unbound operator binds the result (and checks the newcomer)
    @test lattice(Hl + dot(S[1], S[4])) == sites
    @test_throws ArgumentError Hl + dot(S[1], S[5])
    @test_throws ArgumentError Hl + onlattice(dot(S[1], S[2]), fill(su2, 5))
end
