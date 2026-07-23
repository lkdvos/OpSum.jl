using Test
using OpSum
using OpSum: instantiate
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using LinearAlgebra: dot, norm, I as Id

LO(x) = OpSum.LocalOp(x)

# dense reference Pauli / spin-1/2
const σX = ComplexF64[0 1; 1 0]
const σY = ComplexF64[0 -im; im 0]
const σZ = ComplexF64[1 0; 0 -1]

@testset "LocalOp scalar / sum instantiation" begin
    V = SU2Space(1 // 2 => 1)

    # scalar identity → structural c·id(V), NOT via one(::Type)
    Osc = instantiate(scalarop(2.0, V), V)
    @test space(Osc) == (V ← V)
    @test convert(Array, Osc) ≈ 2 * Matrix{ComplexF64}(Id, 2, 2)

    # scalarop can also be built from the sector type
    @test space(instantiate(scalarop(1, SU2Irrep), V)) == (V ← V)

    # bare ITO
    Osp = instantiate(spin(V), V)
    @test space(Osp) == (V ← V ⊗ Vect[SU2Irrep](SU2Irrep(1) => 1))
    @test norm(convert(Array, Osp)) > 0

    # sum (scaled) distributes over instantiate
    S2 = spin(V) + 0.5 * spin(V)
    @test convert(Array, instantiate(S2, V)) ≈ 1.5 * convert(Array, Osp)

    # spin normalization: reduced element √(s(s+1)(2s+1)) = √(3/2) for spin-1/2
    coeff = only(values(OpSum.variant(spin(V)).terms))
    @test coeff ≈ sqrt(3 / 2)

    # on-site products/powers are deferred (build the Prod variant directly, since `*` between
    # two symbolic LocalOps is itself not implemented)
    A = IrrepOperator{SU2Irrep}
    L = OpSum.LocalOp{ComplexF64, A}
    pr = L(OpSum.Prod{L}(L[spin(V), spin(V)]))
    @test_throws ArgumentError instantiate(pr, V)
end

@testset "single-site field embedding" begin
    V = SU2Space(1 // 2 => 1)
    B = convert(Array, instantiate(spin(V), V))          # (2,2,3)
    Wf = convert(Array, instantiate(spin(V)[2], [V, V, V]))
    @test size(Wf) == (2, 2, 2, 2, 2, 2, 3)
    I2 = Matrix{ComplexF64}(Id, 2, 2)
    oracle = [
        I2[o1, i1] * B[o2, i2, q] * I2[o3, i3]
            for o1 in 1:2, o2 in 1:2, o3 in 1:2, i1 in 1:2, i2 in 1:2, i3 in 1:2, q in 1:3
    ]
    @test Wf ≈ oracle

    # empty-site identity is the structural identity on all sites
    idop = one(spin(V)[1])
    @test reshape(convert(Array, instantiate(idop, [V, V])), 4, 4) ≈
        Matrix{Float64}(Id, 4, 4)
end

@testset "SU(2) two-site S·S = Cartesian dot" begin
    # spin-1/2 : ¼(σx⊗σx + σy⊗σy + σz⊗σz)
    V = SU2Space(1 // 2 => 1)
    Sx, Sy, Sz = σX / 2, σY / 2, σZ / 2
    Wd = dropdims(convert(Array, instantiate(dot(spin(V)[1], spin(V)[2]), [V, V])); dims = 5)
    orc = [
        Sx[o1, i1] * Sx[o2, i2] + Sy[o1, i1] * Sy[o2, i2] + Sz[o1, i1] * Sz[o2, i2]
            for o1 in 1:2, o2 in 1:2, i1 in 1:2, i2 in 1:2
    ]
    @test Wd ≈ orc
    # explicit ¼σ·σ form
    quarter = 0.25 * (kron(σX, σX) + kron(σY, σY) + kron(σZ, σZ))
    @test reshape(permutedims(Wd, (2, 1, 4, 3)), 4, 4) ≈ quarter

    # spin-1 Heisenberg
    V1 = SU2Space(1 => 1)
    sq2 = sqrt(2.0)
    Sz1 = ComplexF64[1 0 0; 0 0 0; 0 0 -1]
    Sp1 = ComplexF64[0 sq2 0; 0 0 sq2; 0 0 0]
    Sm1 = collect(Sp1')
    Sx1, Sy1 = (Sp1 + Sm1) / 2, (Sp1 - Sm1) / (2im)
    Wd1 = dropdims(convert(Array, instantiate(dot(spin(V1)[1], spin(V1)[2]), [V1, V1])); dims = 5)
    orc1 = [
        Sx1[o1, i1] * Sx1[o2, i2] + Sy1[o1, i1] * Sy1[o2, i2] + Sz1[o1, i1] * Sz1[o2, i2]
            for o1 in 1:3, o2 in 1:3, i1 in 1:3, i2 in 1:3
    ]
    @test Wd1 ≈ orc1
end

@testset "trivial-sector coupling (ℂ²)" begin
    V = ℂ^2
    ops = instances(IrrepOperator, V)
    E2 = convert(Array, instantiate(ops[2], V))[:, :, 1]
    E3 = convert(Array, instantiate(ops[3], V))[:, :, 1]
    # bare `couple` carries no factor (the Cartesian −√dim lives in `·`/`dot`, not here)
    Cd = dropdims(convert(Array, instantiate(couple(LO(ops[2])[1], LO(ops[3])[2]; to = unit(Trivial)), [V, V])); dims = 5)
    orc = [E2[o1, i1] * E3[o2, i2] for o1 in 1:2, o2 in 1:2, i1 in 1:2, i2 in 1:2]
    @test Cd ≈ orc
end

@testset "U(1) coupling / charge structure" begin
    V = Rep[U₁](0 => 1, 1 => 1)
    raise = LO(IrrepOperator(U1Irrep(1), 1))
    lower = LO(IrrepOperator(U1Irrep(-1), 1))

    # singlet hopping term is number-conserving (net charge 0)
    Hop = instantiate(dot(raise[1], lower[2]), [V, V])
    @test space(Hop) == ((V ⊗ V) ← (V ⊗ V ⊗ Vect[U1Irrep](U1Irrep(0) => 1)))
    rd = dropdims(convert(Array, instantiate(raise, V)); dims = 3)
    ld = dropdims(convert(Array, instantiate(lower, V)); dims = 3)
    Hd = dropdims(convert(Array, Hop); dims = 5)
    orc = [-rd[o1, i1] * ld[o2, i2] for o1 in 1:2, o2 in 1:2, i1 in 1:2, i2 in 1:2]
    @test Hd ≈ orc

    # coupling two charge +1 operators to a definite total charge +2
    C2 = instantiate(couple(raise[1], raise[2]; to = U1Irrep(2)), [V, V])
    @test collect(blocksectors(C2)) == [U1Irrep(2)]
end

@testset "coupling API errors / deferrals" begin
    V = SU2Space(1 // 2 => 1)
    a, b, c = spin(V)[1], spin(V)[2], spin(V)[3]
    # multi-body / tree coupling deferred to Phase 3
    @test_throws ArgumentError couple(a, b; to = SU2Irrep(0), via = :tree)
    @test_throws ArgumentError couple(a, b, c; to = SU2Irrep(0))
    # coupling a scalar (no charge leg) is invalid
    @test_throws ArgumentError instantiate(couple(scalarop(1, V)[1], b; to = SU2Irrep(0)), [V, V, V])
    # same-site coupling invalid
    @test_throws ArgumentError instantiate(couple(spin(V)[1], spin(V)[1]; to = SU2Irrep(0)), [V, V])
end
