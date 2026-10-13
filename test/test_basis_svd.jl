using Test
using OpSum
using OpSum: SVDBondAlgorithm, IndependentSVD, SequentialSVD, Terms
using TensorKit
using LinearAlgebra: norm
import TensorKitTensors.FermionOperators as FO

include("testutils.jl")

# A truncating SVD truncates in the basis' Hilbert–Schmidt metric, so the result depends on the span of
# the basis and not on how it was written.

dense(W) = mpo_tensormap(irrep_mpo_tensors(W))
pairop(ts, B) = plain(instantiate(opsum(FiniteChain(B, 2), ts)))

@testset "the orthonormal letters carry the operator's norm" begin
    # Σ |coeff|² over the orthonormal letters is ‖H‖² / dot(𝟙, 𝟙)^N (SU(2) charge; fermion `n` overlapping 𝟙)
    V = SU2Space(1 // 2 => 1)
    BU = unitbasis(V)
    S = spinvector(BU)
    B = Basis([pairop(dot(S[1], S[2]), BU)])
    ss = project(pairop(dot(S[1], S[2]), BU), B)
    Hs = opsum(FiniteChain(B, 3), ss[1, 2], 0.4 * ss[1, 3], -0.3 * ss[2, 3])
    V2 = FO.fermion_space(Trivial)
    hop, hop′, num = FO.f_plus_f_min(ComplexF64, Trivial), FO.f_min_f_plus(ComplexF64, Trivial),
        FO.f_num(ComplexF64, Trivial)
    F = Basis([hop, hop′, num])
    h, n = project(hop, F), project(num, F)
    Hf = opsum(FiniteChain(F, 3), h[1, 2], 0.5 * h[3, 1], 0.7 * n[2], -0.2 * n[1] * n[3])
    for (H, d) in ((Hs, 2), (Hf, 2))
        metrics = [OpSum._lettermetric(H.lattice[s]) for s in 1:3]
        ts = OpSum._orthonormal_terms(H.terms, metrics)
        @test sum(t -> abs2(t.coeff), ts) ≈ norm(instantiate(H))^2 / d^3
    end
    # an orthonormal basis (the Paulis: dot(σ, σ) = dot(𝟙, 𝟙)) is left alone
    Xm, Ym, Zm = ComplexF64[0 1; 1 0], ComplexF64[0 -im; im 0], ComplexF64[1 0; 0 -1]
    @test OpSum._lettermetric(Basis([Xm, Ym, Zm])) === nothing
end

@testset "a truncation does not depend on how the basis is written" begin
    Im, Xm, Ym, Zm = ComplexF64[1 0; 0 1], ComplexF64[0 1; 1 0], ComplexF64[0 -im; im 0],
        ComplexF64[1 0; 0 -1]
    N = 3
    J = [1 / abs(i - j)^1.3 for i in 1:N, j in 1:N]
    function ising(B)
        x, z = project(Xm, B), project(Zm, B)
        return opsum(
            FiniteChain(B, N),
            (J[i, j] * z[i] * z[j] + 0.3J[i, j] * x[i] * x[j] for i in 1:N for j in (i + 1):N),
            (0.7 * x[i] + 0.2 * z[i] for i in 1:N)
        )
    end
    # the same span, written orthogonally and with overlaps (one of them with the identity)
    H1, H2 = ising(Basis([Xm, Ym, Zm])), ising(Basis([Xm, Xm + Ym, Zm + 0.7Im]))
    D = instantiate(H1)
    @test instantiate(H2) ≈ D
    for sweep in (IndependentSVD, SequentialSVD)
        alg = SVDBondAlgorithm(truncrank(2); sweep)
        A1, A2 = dense(mpo(H1, alg)), dense(mpo(H2, alg))
        @test norm(A1 - D) / norm(D) > 0.05                     # a real truncation
        @test A2 ≈ A1
    end
    @test dense(mpo(H2, SVDBondAlgorithm())) ≈ D               # lossless is unchanged

    # fermions, with `n` against the traceless `n - 1/2`
    V = FO.fermion_space(Trivial)
    hop, hop′, num = FO.f_plus_f_min(ComplexF64, Trivial), FO.f_min_f_plus(ComplexF64, Trivial),
        FO.f_num(ComplexF64, Trivial)
    t = [cos(i + 2j) / (1 + abs(i - j)) for i in 1:N, j in 1:N]
    function chain(B)
        h, n = project(hop, B), project(num, B)
        return opsum(
            FiniteChain(B, N), (t[i, j] * (h[i, j] + h[j, i]) for i in 1:N for j in (i + 1):N),
            (0.4 * n[i] * n[i + 1] for i in 1:(N - 1))
        )
    end
    F1, F2 = chain(Basis([hop, hop′, num])), chain(Basis([hop, hop′, num - 0.5 * id(V)]))
    Df = instantiate(F1)
    @test instantiate(F2) ≈ Df
    alg = SVDBondAlgorithm(truncrank(2))
    A1, A2 = dense(mpo(F1, alg)), dense(mpo(F2, alg))
    @test norm(A1 - Df) / norm(Df) > 0.01
    @test A2 ≈ A1
end
