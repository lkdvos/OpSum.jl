using Test
using OpSum
using OpSum: LocalOperator, mpo_tensormap, irrep_mpo_tensors, jordan_mpo_tensors
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using TensorKit: BraidingTensor

include(joinpath(@__DIR__, "testutils.jl"))

# Chains of per-site bases; the oracle is the same Hamiltonian in the matrix-unit basis `unitbasis(V)`.

const Imat = ComplexF64[1 0; 0 1]
const Xmat = ComplexF64[0 1; 1 0]
const Ymat = ComplexF64[0 -im; im 0]
const Zmat = ComplexF64[1 0; 0 -1]
const Spmat = ComplexF64[0 1; 0 0]

pairop(ts, V, N = 2) = plain(instantiate(opsum(FiniteChain(V, N), ts)))

# a spinless fermion site: `𝟙, n, c†, c` from the two hopping pairs and the number operator
function fermion_site()
    V = Vect[FermionNumber](0 => 1, 1 => 1)
    BU = unitbasis(V)
    F = fermionmode(BU, FermionNumber(0), FermionNumber(1))
    hop, hop′ = pairop(couple(F.cd[1], F.c[2]), BU), pairop(couple(F.c[1], F.cd[2]), BU)
    n = pairop(F.n[1], BU, 1)
    B = Basis([hop, hop′, n])
    return (; V, BU, F, B, h = project(hop, B), h′ = project(hop′, B), n = project(n, B))
end

@testset "a chain of bases" begin
    B = Basis([Xmat, Ymat, Zmat])
    lat = FiniteChain(B, 3)
    @test lat == FiniteChain(fill(B, 3)) && length(lat) == 3 && lat[2] === B
    @test InfiniteChain([B, B])[5] === B
    X = project(Xmat, B)
    @test opsum(fill(B, 2), X[1]).lattice == FiniteChain(B, 2)

    # one kind of site, of one symmetry
    @test_throws ArgumentError FiniteChain([B, ℂ^2])
    @test_throws ArgumentError FiniteChain(ℂ^2, 3)           # a lattice of spaces
    @test_throws ArgumentError FiniteChain([B, fermion_site().B])
    @test_throws ArgumentError InfiniteChain(Basis[])
end

@testset "letters are checked against the site's basis" begin
    small = Basis([Zmat])                                     # 𝟙, Z
    big = Basis([Xmat, Ymat, Zmat])                           # 𝟙, X, Y, Z
    @test opsum(FiniteChain(small, 2), project(Zmat, small)[1]) isa OperatorSum
    # the letter of Y in `big` is (Trivial, 3), which `small` does not have
    @test_throws ArgumentError opsum(FiniteChain(small, 2), project(Ymat, big)[1])
    # a mismatch within range is not caught: X of `big`, (Trivial, 2), reads as Z
    @test opsum(FiniteChain(small, 2), project(Xmat, big)[1]) isa OperatorSum
end

@testset "fermions on a chain of bases: dense, tensors and Jordan form" begin
    f = fermion_site()
    N = 3
    H = opsum(
        FiniteChain(f.B, N),
        (-(f.h[i, i + 1] - f.h′[i, i + 1]) for i in 1:(N - 1)),
        0.4 * f.h[1, 3],
        (0.5 * f.n[i] for i in 1:N),
    )
    F = f.F
    Hm = opsum(
        FiniteChain(f.BU, N),
        (-(couple(F.cd[i], F.c[i + 1]) + couple(F.cd[i + 1], F.c[i])) for i in 1:(N - 1)),
        0.4 * couple(F.cd[1], F.c[3]),
        (0.5 * F.n[i] for i in 1:N),
    )
    @test instantiate(H) ≈ instantiate(Hm)
    @test islossless(H)
    @test mpo_tensormap(irrep_mpo_tensors(mpo(H))) ≈ instantiate(H)
    Ws = jordan_mpo_tensors(H)
    @test Ws[2][1, 1, 1, 1] isa BraidingTensor
    @test mpo_tensormap(map(TensorMap, Ws)) ≈ instantiate(H)
end

@testset "H' projects back into the site bases" begin
    f = fermion_site()
    lat = FiniteChain(f.B, 2)
    H = opsum(lat, f.h[1, 2], (0.3 + 0.2im) * f.n[2])
    # (c†₁c₂)† = c†₂c₁ = -c₁c†₂
    @test H' ≈ opsum(lat, -f.h′[1, 2], (0.3 - 0.2im) * f.n[2])
    @test plain(instantiate(H')) ≈ plain(instantiate(H))'

    # a basis that is not closed under adjoints: S⁺ is atomic, S⁻ is not in the span
    A = Basis([Spmat])
    @test_throws ArgumentError opsum(FiniteChain(A, 1), project(Spmat, A)[1])'
end

@testset "mixed sites: vcat of bases on different spaces" begin
    Lz = ComplexF64[1 0 0; 0 0 0; 0 0 -1]
    P2 = Basis([Xmat, Ymat, Zmat])
    P3 = Basis([Lz])
    # Z on a qubit times Lz on a qutrit, one term in the per-slot bases
    ZL = TensorMap(kron(Lz, Zmat), ℂ^2 ⊗ ℂ^3 ← ℂ^2 ⊗ ℂ^3)            # column-major: slot 1 is Z
    zl = project(ZL, [P2, P3])
    @test zl isa LocalOperator && length(zl.terms) == 1
    lat = FiniteChain(vcat(P2, P3, P2))
    H = opsum(lat, zl[1, 2], 0.5 * project(Xmat, P2)[3])

    U2, U3 = unitbasis(ℂ^2), unitbasis(ℂ^3)
    units = FiniteChain(vcat(U2, U3, U2))
    Zm = project(TensorMap(Zmat, ℂ^2 ← ℂ^2), U2)
    Xm = project(TensorMap(Xmat, ℂ^2 ← ℂ^2), U2)
    Lm = project(TensorMap(Lz, ℂ^3 ← ℂ^3), U3)
    Hm = opsum(units, Zm[1] * Lm[2], 0.5 * Xm[3])
    @test instantiate(H) ≈ instantiate(Hm)
    @test mpo_tensormap(irrep_mpo_tensors(mpo(H))) ≈ instantiate(H)
    @test_throws ArgumentError project(ZL, [P2, P2])                 # slot 2 is a qutrit
end

@testset "infinite chain of bases" begin
    P = Basis([Xmat, Ymat, Zmat])
    Os = [project(M, P) for M in (Xmat, Ymat, Zmat)]
    H = opsum(InfiniteChain(P), (couple(O[1], O[2]) for O in Os))
    Hinf = mpo(H)
    @test sum(dim, Hinf.bondsectors[1]) == 5                       # 3 channels + start + done
    Ts = irrep_mpo_tensors(Hinf)
    @test space(Ts[1], 1) == space(Ts[1], 4)'
end

@testset "decaying channels over bases" begin
    # a decaying hop with a parity string, Σ_{i<j} λ^{j-i-1} c†ᵢ (Π_k Pₖ) cⱼ + n (N = 3 is the dense size used throughout the file)
    f = fermion_site()
    λ, N = 0.6, 3
    Pm = id(f.V) - 2 * pairop(f.F.n[1], f.BU, 1)
    p = project(Pm, f.B)
    lat = FiniteChain(f.B, N)
    H = opsum(lat, expterm(f.h[1, 2]; decay = λ, string = p), 0.3 * f.n[2])
    want = opsum(
        lat, 0.3 * f.n[2],
        (λ^(j - i - 1) * foldl(*, (p[k] for k in (i + 1):(j - 1)); init = f.h[i, j]) for i in 1:N for j in (i + 1):N)
    )
    @test instantiate(H) ≈ instantiate(want)
    @test mpo_tensormap(irrep_mpo_tensors(mpo(H))) ≈ instantiate(want)

    # the same on an infinite chain, against the explicit window
    Hi = opsum(InfiniteChain(f.B), expterm(f.h[1, 2]; decay = λ, string = p), 0.3 * f.n[1])
    W = mpo(Hi)
    got = Dict(t => t.coeff for t in OpSum.mpo_terms_window(W, 6))
    exact = Dict(t => t.coeff for t in OpSum.window_terms(Hi, 6))
    @test all(haskey(exact, t) && exact[t] ≈ v for (t, v) in got)
    @test all(haskey(got, t) for t in keys(exact) if maximum(t.sites) <= 4)

    # a channel's letters land on sites it does not name: they must mean the same operator there
    Q = Basis([Xmat, Zmat])
    x, z = project(Xmat, Q), project(Zmat, Q)
    Z1 = Basis([Zmat])
    @test_throws ArgumentError opsum(FiniteChain(vcat(Q, Z1, Q)), expterm(z[1] * z[2]; decay = λ))
    z1 = project(Zmat, Z1)
    @test opsum(InfiniteChain([Q, Z1]), expterm(z[1] * z1[2]; decay = λ)) isa OperatorSum
    @test_throws ArgumentError opsum(InfiniteChain([Q, Z1]), expterm(z[1] * z[3]; decay = λ, string = z))
    # `z` is letter (unit, 3) of `Q`, which `Z1` lacks
    @test_throws ArgumentError opsum(InfiniteChain(Z1), expterm(z1[1] * z1[2]; decay = λ, string = z))
end
