using Test
using OpSum
using OpSum: ITOTermTable, _op_at_ito, arity, nterms, nvertices, ispassthrough,
    TermSum, spin, scalarop, couple
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
