using Test
using OpSum
using OpSum: irrep_mpo, irrep_mpo_tensors, instantiate, spin, scalarop, couple
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using TensorKit: @tensor

include(joinpath(@__DIR__, "testutils.jl"))   # LO, physmatrix

@testset "SU(2) MPO tensors contract to the operator" begin
    V = SU2Space(1 // 2 => 1)
    d = 2

    # --- N = 2 : single S·S term ---
    H2 = dot(spin(V)[1], spin(V)[2])
    Ws, secs = irrep_mpo(H2, [V, V])
    T = irrep_mpo_tensors(Ws, secs, [V, V])
    @tensor Op2[o1 o2 bL; bR i1 i2] := T[1][bL o1; i1 bm] * T[2][bm o2; i2 bR]

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
        T3[1][bL o1; i1 b1] * T3[2][b1 o2; i2 b2] * T3[3][b2 o3; i3 bR]

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
        T[1][bL o1; i1 b1] * T[2][b1 o2; i2 b2] * T[3][b2 o3; i3 bR]
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
    @tensor Op[o1 o2 bL; bR i1 i2] := T[1][bL o1; i1 bm] * T[2][bm o2; i2 bR]
    @test physmatrix(Op, 2, d) ≈ physmatrix(instantiate(H, fill(V, 2)), 2, d)
end

@testset "K=3 SU(2) MPO tensors contract to the operator" begin
    V = SU2Space(1 // 2 => 1)
    d = 2
    S = spin(V)
    # genuine 3-body scalar: ((S₁⊗S₂)→spin-1) ⊗ S₃ → singlet
    H = couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0))
    Ws, secs = irrep_mpo(H, fill(V, 3))
    T = irrep_mpo_tensors(Ws, secs, fill(V, 3))
    @tensor Op[o1 o2 o3 bL; bR i1 i2 i3] :=
        T[1][bL o1; i1 b1] * T[2][b1 o2; i2 b2] * T[3][b2 o3; i3 bR]
    @test physmatrix(Op, 3, d) ≈ physmatrix(instantiate(H, fill(V, 3)), 3, d)
end

# Physical cross-check of the coupling convention: the unique SU(2)-invariant 3-spin scalar in the
# ((S₁⊗S₂)→1)⊗S₃→0 channel is proportional to the scalar spin chirality S₁·(S₂×S₃).
@testset "K=3 SU(2) scalar ∝ spin chirality" begin
    V = SU2Space(1 // 2 => 1)
    S = spin(V)
    Sx = ComplexF64[0 1; 1 0] / 2
    Sy = ComplexF64[0 -im; im 0] / 2
    Sz = ComplexF64[1 0; 0 -1] / 2
    Sv = (Sx, Sy, Sz)

    eps = zeros(3, 3, 3)
    eps[1, 2, 3] = eps[2, 3, 1] = eps[3, 1, 2] = 1
    eps[1, 3, 2] = eps[3, 2, 1] = eps[2, 1, 3] = -1
    chi = zeros(ComplexF64, 8, 8)                     # χ = Σ ε_abc S₁ᵃ S₂ᵇ S₃ᶜ  (kron site order)
    for a in 1:3, b in 1:3, c in 1:3
        iszero(eps[a, b, c]) && continue
        chi += eps[a, b, c] * kron(Sv[a], kron(Sv[b], Sv[c]))
    end

    H = couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0))
    Hmat = physmatrix(instantiate(H, fill(V, 3)), 3, 2)
    α = dot(vec(chi), vec(Hmat)) / dot(vec(chi), vec(chi))   # least-squares proportionality constant
    @test abs(α) > 1.0e-6
    @test Hmat ≈ α * chi
end
