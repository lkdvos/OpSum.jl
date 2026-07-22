using Test
using OpSum
using OpSum: irrep_mpo, irrep_mpo_tensors, instantiate, spin, scalarop, couple
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using TensorKit: @tensor

LO(x) = OpSum.LocalOp(x)

# densify a TensorMap operator to a matrix: drop dim-1 (boundary/charge) legs, keep the 2N
# physical axes ordered [out_1..N, in_1..N], reorder to kron index convention.
function physmatrix(t, N, d)
    A = convert(Array, t)
    A = dropdims(A; dims = Tuple(findall(==(1), size(A))))
    @assert ndims(A) == 2N
    perm = (reverse(1:N)..., reverse((N + 1):(2N))...)
    return reshape(permutedims(A, perm), d^N, d^N)
end

@testset "SU(2) MPO tensors contract to the operator" begin
    V = SU2Space(1 // 2 => 1)
    d = 2

    # --- N = 2 : single S·S term ---
    H2 = dot(spin(V)[1], spin(V)[2])
    Ws, secs = irrep_mpo(H2, [V, V])
    T = irrep_mpo_tensors(Ws, secs, [V, V])
    @tensor Op2[o1 o2 bL; bR i1 i2] := T[1][o1 bL; bm i1] * T[2][o2 bm; bR i2]

    M_mpo = physmatrix(Op2, 2, d)
    M_ref = physmatrix(instantiate(H2, [V, V]), 2, d)
    @test M_mpo ≈ M_ref

    # physical sanity: equals ¼(σx⊗σx + σy⊗σy + σz⊗σz)
    σX = ComplexF64[0 1; 1 0]; σY = ComplexF64[0 -im; im 0]; σZ = ComplexF64[1 0; 0 -1]
    @test M_mpo ≈ 0.25 * (kron(σX, σX) + kron(σY, σY) + kron(σZ, σZ))

    # --- N = 3 : Heisenberg chain ---
    H3 = dot(spin(V)[1], spin(V)[2]) + dot(spin(V)[2], spin(V)[3])
    Ws3, secs3 = irrep_mpo(H3, fill(V, 3))
    T3 = irrep_mpo_tensors(Ws3, secs3, fill(V, 3))
    @tensor Op3[o1 o2 o3 bL; bR i1 i2 i3] :=
        T3[1][o1 bL; b1 i1] * T3[2][o2 b1; b2 i2] * T3[3][o3 b2; bR i3]

    @test physmatrix(Op3, 3, d) ≈ physmatrix(instantiate(H3, fill(V, 3)), 3, d)
end

@testset "trivial sector (ℂ²) MPO tensors contract to the operator" begin
    V = ℂ^2
    d = 2
    ops = instances(IrrepOperator, V)
    H = couple(LO(ops[2])[1], LO(ops[3])[2]; to = unit(Trivial)) +
        couple(LO(ops[2])[2], LO(ops[3])[3]; to = unit(Trivial))
    Ws, secs = irrep_mpo(H, fill(V, 3))
    T = irrep_mpo_tensors(Ws, secs, fill(V, 3))
    @tensor Op[o1 o2 o3 bL; bR i1 i2 i3] :=
        T[1][o1 bL; b1 i1] * T[2][o2 b1; b2 i2] * T[3][o3 b2; bR i3]
    @test physmatrix(Op, 3, d) ≈ physmatrix(instantiate(H, fill(V, 3)), 3, d)
end

@testset "U(1) hopping MPO tensors contract to the operator" begin
    V = Rep[U₁](0 => 1, 1 => 1)
    d = 2
    raise = LO(IrrepOperator(U1Irrep(1), 1))
    lower = LO(IrrepOperator(U1Irrep(-1), 1))
    H = dot(raise[1], lower[2]) + dot(lower[1], raise[2])
    Ws, secs = irrep_mpo(H, fill(V, 2))
    T = irrep_mpo_tensors(Ws, secs, fill(V, 2))
    @tensor Op[o1 o2 bL; bR i1 i2] := T[1][o1 bL; bm i1] * T[2][o2 bm; bR i2]
    @test physmatrix(Op, 2, d) ≈ physmatrix(instantiate(H, fill(V, 2)), 2, d)
end
