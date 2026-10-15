using Test
using OpSum
using OpSum: mpo, mpo_tensors, instantiate, scalarop, couple, opsum
using OpSum: Letter
using TensorKit
using TensorKit: @tensor

include(joinpath(@__DIR__, "..", "testutils.jl"))   # LO, physmatrix

contract2(T) = @tensor Op[o1 o2 bL; bR i1 i2] := T[1][bL o1; i1 bm] * T[2][bm o2; i2 bR]
function contract3(T)
    return @tensor Op[o1 o2 o3 bL; bR i1 i2 i3] :=
        T[1][bL o1; i1 b1] * T[2][b1 o2; i2 b2] * T[3][b2 o3; i3 bR]
end

@testset "MPO tensors contract to the operator" begin
    Vs = SU2Space(1 // 2 => 1)
    S = spinvector(unitbasis(Vs))
    Vu = Rep[U₁](0 => 1, 1 => 1)
    raise, lower = LO(Letter(U1Irrep(1), 1)), LO(Letter(U1Irrep(-1), 1))
    ops = [LO(Letter(unit(Trivial), n)) for n in 1:4]
    cases = [
        ("SU2 S·S", unitbasis(Vs), 2, opsum(dot(S[1], S[2]))),
        ("SU2 Heisenberg", unitbasis(Vs), 3, opsum(dot(S[1], S[2]), dot(S[2], S[3]))),
        ("trivial", unitbasis(ℂ^2), 3, opsum(couple(ops[2][i], ops[3][i + 1]; to = unit(Trivial)) for i in 1:2)),
        ("U1 hopping", unitbasis(Vu), 2, opsum(dot(raise[1], lower[2]), dot(lower[1], raise[2]))),
        ("SU2 K=3", unitbasis(Vs), 3, opsum(couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0)))),
    ]
    for (name, B, N, H) in cases
        sites = fill(B, N)
        T = mpo_tensors(reduced(mpo(opsum(sites, H)))..., sites)
        M = physmatrix(N == 2 ? contract2(T) : contract3(T), N, 2)
        @test M ≈ physmatrix(instantiate(opsum(sites, H)), N, 2)
        if name == "SU2 S·S"
            σX = ComplexF64[0 1; 1 0]; σY = ComplexF64[0 -im; im 0]; σZ = ComplexF64[1 0; 0 -1]
            @test M ≈ 0.25 * (kron(σX, σX) + kron(σY, σY) + kron(σZ, σZ))
        end
    end
end

# Physical cross-check of the coupling convention: the unique SU(2)-invariant 3-spin scalar in the
# ((S₁⊗S₂)→1)⊗S₃→0 channel is proportional to the scalar spin chirality S₁·(S₂×S₃).
@testset "K=3 SU(2) scalar ∝ spin chirality" begin
    V = SU2Space(1 // 2 => 1)
    S = spinvector(unitbasis(V))
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

    H = opsum(couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0)))
    Hmat = physmatrix(instantiate(opsum(fill(unitbasis(V), 3), H)), 3, 2)
    α = dot(vec(chi), vec(Hmat)) / dot(vec(chi), vec(chi))   # least-squares proportionality constant
    @test abs(α) > 1.0e-6
    @test Hmat ≈ α * chi
end
