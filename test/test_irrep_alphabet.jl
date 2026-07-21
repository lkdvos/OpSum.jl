using Test
using OpSum
using OpSum: instantiate
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using VectorInterface: inner, scalartype
using LinearAlgebra: tr, rank, norm

# Reference (dense) Pauli / spin matrices
const σI = ComplexF64[1 0; 0 1]
const σX = ComplexF64[0 1; 1 0]
const σY = ComplexF64[0 -im; im 0]
const σZ = ComplexF64[1 0; 0 -1]

@testset "SU(2) spin-½" begin
    V = SU2Space(1 // 2 => 1)
    ops = instances(IrrepOperator, V)

    @test ops == [IrrepOperator(SU2Irrep(0), 1), IrrepOperator(SU2Irrep(1), 1)]
    @test length(ops) == 2

    # identity element (c = 0): normalized identity
    id_op = one(IrrepOperator, V)
    @test id_op == IrrepOperator(SU2Irrep(0), 1)
    A = convert(Array, instantiate(id_op, V))
    @test size(A) == (2, 2, 1)
    @test A[:, :, 1] ≈ σI ./ sqrt(2)

    # charge-1 element: the spin / vector operator (rank-1 spherical tensor)
    B = convert(Array, instantiate(IrrepOperator(SU2Irrep(1), 1), V))
    @test size(B) == (2, 2, 3)          # 3 = dim of the spin-1 coupled leg
    for q in 1:3                         # each spherical component is traceless
        @test abs(tr(B[:, :, q])) < 1.0e-12
    end
    # the three components are linearly independent and span the traceless
    # 2×2 operators — i.e. the same space as {σx, σy, σz}
    comps = hcat((vec(B[:, :, q]) for q in 1:3)...)
    @test rank(comps) == 3
    @test rank(hcat(comps, vec(σX), vec(σY), vec(σZ))) == 3
end

@testset "trivial sector ℂ²" begin
    V = ℂ^2
    ops = instances(IrrepOperator, V)
    @test length(ops) == 4               # matrix units spanning End(ℂ²)

    mats = [convert(Array, instantiate(op, V))[:, :, 1] for op in ops]
    @test rank(hcat(vec.(mats)...)) == 4  # linearly independent

    # each alphabet element is normalized under TensorKit's inner
    for op in ops
        t = instantiate(op, V)
        @test inner(t, t) ≈ 1
    end

    # projecting an arbitrary 2×2 operator preserves the norm (orthonormal basis)
    for M in (σX, σY, σZ, σI)
        coeffs = [sum(conj.(E) .* M) for E in mats]   # ⟨E, M⟩_Frobenius
        recon = sum(c .* E for (c, E) in zip(coeffs, mats))
        @test recon ≈ M
        @test norm(coeffs) ≈ norm(M)
    end
end

@testset "U(1) toy space" begin
    V = Rep[U₁](0 => 1, 1 => 1)
    ops = instances(IrrepOperator, V)

    @test Set(op.c for op in ops) == Set([U1Irrep(-1), U1Irrep(0), U1Irrep(1)])
    @test length(ops) == 4               # abelian: reduceddim² = 2²

    # the raising element (charge +1) maps the charge-0 subspace to charge-1
    raise = instantiate(IrrepOperator(U1Irrep(1), 1), V)
    @test collect(blocksectors(raise)) == [U1Irrep(1)]
    # the lowering element (charge -1) lands in block sector 0
    lower = instantiate(IrrepOperator(U1Irrep(-1), 1), V)
    @test collect(blocksectors(lower)) == [U1Irrep(0)]
end

# spaces exercising multiple sectors and internal degeneracy
const SPACES = (
    SU2Space(1 // 2 => 1),
    ℂ^2,
    Rep[U₁](0 => 1, 1 => 1),
    SU2Space(1 // 2 => 1, 3 // 2 => 1),
    SU2Space(1 // 2 => 2),
)

@testset "orthonormality — $V" for V in SPACES
    ops = instances(IrrepOperator, V)
    tensors = [instantiate(op, V) for op in ops]
    for i in eachindex(ops), j in eachindex(ops)
        # TensorMap `inner` is only defined between operators of equal charge
        # (same HomSpace); cross-charge orthogonality is structural.
        if ops[i].c == ops[j].c
            @test inner(tensors[i], tensors[j]) ≈ (i == j) atol = 1.0e-12
            # the symbolic inner reproduces the qdim-weighted TensorMap inner
            @test inner(ops[i], ops[j]) ≈ inner(tensors[i], tensors[j]) atol = 1.0e-12
        end
        # the symbolic inner is δᵢⱼ across the whole alphabet
        @test inner(ops[i], ops[j]) ≈ (i == j)
    end
end

@testset "completeness / count" begin
    # universal identity: Σ over the alphabet of dim(c) == dim(V)²
    for V in SPACES
        ops = instances(IrrepOperator, V)
        @test sum(dim(op.c) for op in ops) == dim(V)^2
    end

    # `length == reduceddim(V)²` holds for ABELIAN sectors (one fusion channel per (j,j′))
    for V in (ℂ^2, Rep[U₁](0 => 1, 1 => 1), Rep[U₁](0 => 2, 1 => 1))
        @test length(instances(IrrepOperator, V)) == reduceddim(V)^2
    end
    # non-abelian: the count is the number of fusion channels, larger than reduceddim²
    V = SU2Space(1 // 2 => 1)
    @test length(instances(IrrepOperator, V)) == 2
    @test length(instances(IrrepOperator, V)) != reduceddim(V)^2
end

@testset "ordering, hashing, scalars" begin
    V = SU2Space(1 // 2 => 1)
    ops = instances(IrrepOperator, V)
    @test issorted(ops)
    @test ops[1] < ops[2]

    d = Dict(op => i for (i, op) in enumerate(ops))
    @test d[IrrepOperator(SU2Irrep(1), 1)] == 2
    @test IrrepOperator(SU2Irrep(0), 1) == IrrepOperator(SU2Irrep(0), 1)
    @test IrrepOperator(SU2Irrep(0), 1) != IrrepOperator(SU2Irrep(1), 1)
    @test hash(IrrepOperator(SU2Irrep(0), 1)) == hash(IrrepOperator(SU2Irrep(0), 1))

    @test scalartype(IrrepOperator) == ComplexF64
    @test !isreal(IrrepOperator)
end
