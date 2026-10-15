using Test
using OpSum
using OpSum: mpo, scalarop, couple, opsum
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using LinearAlgebra: dot

include(joinpath(@__DIR__, "..", "testutils.jl"))   # LO, islossless

# Default-algorithm `mpo` is lossless and ends on the term's total charge. The pinned bond profiles of
# the same models live in compression/test_graph.jl.
@testset "mpo is lossless with the right boundary at the total charge" begin
    Vs = SU2Space(1 // 2 => 1)
    S = spinvector(unitbasis(Vs))
    Vu = Rep[U₁](0 => 1, 1 => 1)
    raise, lower = LO(IrrepOperator(U1Irrep(1), 1)), LO(IrrepOperator(U1Irrep(-1), 1))
    ops = [LO(IrrepOperator(unit(Trivial), n)) for n in 1:4]
    heis(N) = opsum(dot(S[i], S[i + 1]) for i in 1:(N - 1))
    cases = Any[
        ("SU2 Heisenberg N=$N", unitbasis(Vs), N, heis(N), SU2Irrep(0)) for N in (3, 4, 5, 6)
    ]
    append!(
        cases, [
            ("SU2 scalar + coupling", unitbasis(Vs), 3, opsum(scalarop(2.5, unitbasis(Vs))[1], dot(S[1], S[2])), SU2Irrep(0)),
            ("SU2 fields", unitbasis(Vs), 3, opsum(S[1], S[2], S[3]), SU2Irrep(1)),
            ("SU2 K=3", unitbasis(Vs), 3, opsum(couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0))), SU2Irrep(0)),
            ("SU2 decoupled pairs", unitbasis(Vs), 6, opsum(dot(S[1], S[2]), dot(S[3], S[4]), dot(S[5], S[6])), SU2Irrep(0)),
            (
                "U1 hopping", unitbasis(Vu), 4,
                opsum((dot(raise[i], lower[i + 1]) for i in 1:3), (dot(lower[i], raise[i + 1]) for i in 1:3)), U1Irrep(0),
            ),
            ("U1 K=3", unitbasis(Vu), 3, opsum(couple(couple(raise[1], raise[2]; to = U1Irrep(2)), lower[3]; to = U1Irrep(1))), U1Irrep(1)),
            (
                "trivial", unitbasis(ℂ^2), 3,
                opsum(couple(ops[2][i], ops[3][i + 1]; to = unit(Trivial)) for i in 1:2), unit(Trivial),
            ),
        ]
    )
    for (name, B, N, H, total) in cases
        @testset "$name" begin
            @test islossless(opsum(fill(B, N), H))
            _, secs = reduced(mpo(opsum(fill(B, N), H)))
            @test secs[N] == [total]
        end
    end

    # charge content of inner bonds not pinned in compression/test_graph.jl
    _, secs = reduced(mpo(opsum(fill(unitbasis(Vs), 6), heis(6))))
    @test count(==(SU2Irrep(0)), secs[3]) == 2      # identity-in + identity-out
    @test count(==(SU2Irrep(1)), secs[3]) == 1      # one open spin multiplet
    H3 = opsum(couple(couple(raise[1], raise[2]; to = U1Irrep(2)), lower[3]; to = U1Irrep(1)))
    _, secs = reduced(mpo(opsum(fill(unitbasis(Vu), 3), H3)))
    @test U1Irrep(2) in secs[2]                     # the inner-line charge
end
