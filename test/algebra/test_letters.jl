using Test
using OpSum
using OpSum: instantiate
using OpSum: Letter
using TensorKit
using VectorInterface: inner, scalartype
using LinearAlgebra: tr, rank

const σI = ComplexF64[1 0; 0 1]
const σX = ComplexF64[0 1; 1 0]
const σY = ComplexF64[0 -im; im 0]
const σZ = ComplexF64[1 0; 0 -1]

@testset "SU(2) spin-½" begin
    V = SU2Space(1 // 2 => 1)
    # index 1 of the trivial charge is the identity, so its matrix units start at 2
    @test OpSum.matrixunits(V) == [Letter(SU2Irrep(0), 2), Letter(SU2Irrep(1), 1)]
    A = convert(Array, OpSum.matrixunit(Letter(SU2Irrep(0), 2), V))
    @test A[:, :, 1] ≈ σI ./ sqrt(2)
    @test convert(Array, OpSum.matrixunit(Letter(SU2Irrep(0), 1), V))[:, :, 1] ≈ σI

    B = convert(Array, OpSum.matrixunit(Letter(SU2Irrep(1), 1), V))   # rank-1 spherical tensor
    @test size(B) == (2, 2, 3)
    @test all(q -> abs(tr(B[:, :, q])) < 1.0e-12, 1:3)
    comps = hcat((vec(B[:, :, q]) for q in 1:3)...)
    @test rank(hcat(comps, vec(σX), vec(σY), vec(σZ))) == 3 == rank(comps)
end

@testset "U(1): raising and lowering letters" begin
    V = Rep[U₁](0 => 1, 1 => 1)
    @test collect(blocksectors(OpSum.matrixunit(Letter(U1Irrep(1), 1), V))) == [U1Irrep(1)]
    @test collect(blocksectors(OpSum.matrixunit(Letter(U1Irrep(-1), 1), V))) == [U1Irrep(0)]
end

const SPACES = (
    SU2Space(1 // 2 => 1),
    ℂ^2,
    Rep[U₁](0 => 1, 1 => 1),
    SU2Space(1 // 2 => 1, 3 // 2 => 1),
    SU2Space(1 // 2 => 2),
)

@testset "orthonormality and completeness — $V" for V in SPACES
    ops = OpSum.matrixunits(V)
    tensors = [OpSum.matrixunit(op, V) for op in ops]
    @test sum(dim(op.c) for op in ops) == dim(V)^2
    for i in eachindex(ops), j in eachindex(ops)
        # TensorMap `inner` needs equal charge; cross-charge orthogonality is structural
        if ops[i].c == ops[j].c
            @test inner(tensors[i], tensors[j]) ≈ (i == j) atol = 1.0e-12
            @test inner(ops[i], ops[j]) ≈ inner(tensors[i], tensors[j]) atol = 1.0e-12
        end
        @test inner(ops[i], ops[j]) ≈ (i == j)
    end
end

@testset "abelian count is reduceddim², non-abelian is the number of channels" begin
    for V in (ℂ^2, Rep[U₁](0 => 1, 1 => 1), Rep[U₁](0 => 2, 1 => 1))
        @test length(OpSum.matrixunits(V)) == reduceddim(V)^2
    end
    V = SU2Space(1 // 2 => 1)
    @test length(OpSum.matrixunits(V)) == 2 != reduceddim(V)^2
end

@testset "ordering, hashing, scalars" begin
    ops = OpSum.matrixunits(SU2Space(1 // 2 => 1))
    @test issorted(ops) && ops[1] < ops[2]
    @test Letter(SU2Irrep(0), 1) == Letter(SU2Irrep(0), 1)
    @test Letter(SU2Irrep(0), 1) != Letter(SU2Irrep(1), 1)
    @test hash(Letter(SU2Irrep(0), 1)) == hash(Letter(SU2Irrep(0), 1))
    @test scalartype(Letter) == ComplexF64
    @test !isreal(Letter)
end
