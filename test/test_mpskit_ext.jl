using Test
using OpSum
using OpSum: LocalOperator, mpo_tensormap, irrep_mpo_tensors, jordan_mpo_tensors, mpo_terms
using TensorKit
using TensorKit: removeunit, numind, BraidingTensor
using MPSKit: MPSKit, FiniteMPOHamiltonian, InfiniteMPOHamiltonian, FiniteMPS, InfiniteMPS,
    expectation_value
using TensorKitTensors.HubbardOperators: HubbardOperators as HO
using TensorKitTensors.FermionOperators: FermionOperators as FO
using Random: Random

include("testutils.jl")

# `instantiate(W::SymbolicMPO)` (ext/OpSumMPSKitExt.jl) against MPSKit's own construction from local
# operators, compared on random states. Models use TensorKitTensors operators projected into bases built from them.

Random.seed!(1234)


# `h` on sites (1, 3) of a 3-site block; `permute` carries the fermionic sign
gap3(h2, V) = permute(h2 ⊗ id(V), ((1, 3, 2), (4, 6, 5)))

centre(secs, b) = sum(dim, secs[b])

@testset "without MPSKit's methods the fallback says what to load" begin
    @test_throws ArgumentError OpSum._mpskit_hamiltonian(nothing)
end

# One spinless-fermion site (parity only) shared by the testsets, so each MPSKit method compiles for
# one sector type
P = Trivial
V = FO.fermion_space(P)
Vmps = Vect[FermionParity](0 => 4, 1 => 4)
hop, hop′ = FO.f_plus_f_min(ComplexF64, P), FO.f_min_f_plus(ComplexF64, P)
num = FO.f_num(ComplexF64, P)
B = Basis([hop, hop′, num])
h, h′, n = project(hop, B), project(hop′, B), project(num, B)
t, t2, μ = 1.0, 0.4, 0.3
hh = -t * (hop - hop′)                         # hermitian: c†ᵢcⱼ + c†ⱼcᵢ = c†ᵢcⱼ - cᵢc†ⱼ
pair(i, j) = -(h[i, j] - h′[i, j])

@testset "spinless fermions with a hop across a gap, finite and infinite" begin
    # parity only: with particle-number symmetry a random infinite MPS has the wrong filling, and
    # every expectation value vanishes
    @test length(h.terms) == 1 && length(h′.terms) == 1 && length(n) == 1
    N = 4
    H = opsum(
        FiniteChain(B, N),
        (t * pair(i, i + 1) for i in 1:(N - 1)),
        (t2 * pair(i, i + 2) for i in 1:(N - 2)),
        (μ * n[i] for i in 1:N),
    )
    W = mpo(H)
    @test W isa SymbolicMPO
    Hmps = instantiate(W)
    @test Hmps isa FiniteMPOHamiltonian
    ref = FiniteMPOHamiltonian(
        fill(V, N),
        ((i, i + 1) => hh for i in 1:(N - 1))...,
        ((i, i + 1, i + 2) => (t2 / t) * gap3(hh, V) for i in 1:(N - 2))...,
        ((i,) => μ * num for i in 1:N)...,
    )
    ψ = FiniteMPS(randn, ComplexF64, N, V, Vmps)
    @test expectation_value(ψ, Hmps) ≈ expectation_value(ψ, ref)
    @test abs(expectation_value(ψ, ref)) > 1.0e-3

    Hi = opsum(InfiniteChain(B), t * pair(1, 2), t2 * pair(1, 3), μ * n[1])
    Wi = mpo(Hi)
    Hmpsi = instantiate(Wi)
    @test Hmpsi isa InfiniteMPOHamiltonian
    Ts = jordan_mpo_tensors(Wi)
    @test Ts[1][1, 1, 1, 1] isa BraidingTensor
    @test Ts[1][end, 1, 1, end] isa BraidingTensor
    refi = InfiniteMPOHamiltonian([V], (1, 2) => hh, (1, 2, 3) => (t2 / t) * gap3(hh, V), (1,) => μ * num)
    ψi = InfiniteMPS(V, Vmps)
    @test expectation_value(ψi, Hmpsi) ≈ expectation_value(ψi, refi)
    @test abs(expectation_value(ψi, refi)) > 1.0e-3
end

# A Hubbard site with the hopping pieces (spinors under SU(2)), `n` and `nꜛꜜ` as elements.
function hubbard_site(P, S)
    V = HO.hubbard_space(P, S)
    hop, hop′ = HO.e_plus_e_min(ComplexF64, P, S), HO.e_min_e_plus(ComplexF64, P, S)
    num, ud = HO.e_num(ComplexF64, P, S), HO.ud_num(ComplexF64, P, S)
    B = Basis([hop, hop′, num, ud])
    ehop = HO.e_hopping(ComplexF64, P, S)
    return (; V, B, ehop, ud, e = project(ehop, B), u = project(ud, B))
end

@testset "Hubbard chains: bond dimension in the right basis, and energies" begin
    t, U = 1.0, 4.0
    for (P, S, N, today, now) in ((U1Irrep, U1Irrep, 4, 10, 6), (U1Irrep, SU2Irrep, 4, 10, 6))
        site = hubbard_site(P, S)
        H = opsum(FiniteChain(site.B, N), (-t * site.e[i] for i in 1:(N - 1)), (U * site.u[i] for i in 1:N))
        @test islossless(H)
        b = N ÷ 2
        @test centre(mpo(H).bondsectors, b) == now
        BU = unitbasis(site.V)
        ehopm, udm = project(site.ehop, BU), project(site.ud, BU)
        Hm = opsum(
            FiniteChain(BU, N),
            (-t * ehopm[i] for i in 1:(N - 1)),
            (U * udm[i] for i in 1:N),
        )
        @test centre(mpo(Hm).bondsectors, b) == today

        # the MPSKit comparison runs on the non-abelian site (the abelian fermionic one is the testset above)
        S isa SU2Irrep || continue
        Hmps = instantiate(mpo(H))
        ref = FiniteMPOHamiltonian(
            fill(site.V, N), ((i, i + 1) => -t * site.ehop for i in 1:(N - 1))...,
            ((i,) => U * site.ud for i in 1:N)...,
        )
        ψ = FiniteMPS(randn, ComplexF64, N, site.V, oneunit(site.V) ⊕ site.V ⊕ fuse(site.V ⊗ site.V))
        @test expectation_value(ψ, Hmps) ≈ expectation_value(ψ, ref)
    end
end

@testset "spinful all-to-all hopping: 42 in matrix units, 22 in the right basis" begin
    site = hubbard_site(U1Irrep, U1Irrep)
    N = 10
    tij = randn(N, N)
    pairs_ = [(i, j) for i in 1:N for j in (i + 1):N]
    H = opsum(FiniteChain(site.B, N), (tij[i, j] * site.e[i, j] for (i, j) in pairs_))
    @test islossless(H)
    @test centre(mpo(H).bondsectors, N ÷ 2) == 22
    BU = unitbasis(site.V)
    ehop = project(site.ehop, BU)
    Hm = opsum(FiniteChain(BU, N), (tij[i, j] * ehop[i, j] for (i, j) in pairs_))
    @test centre(mpo(Hm).bondsectors, N ÷ 2) == 42
end

@testset "the SymbolicMPO itself" begin
    N = 4
    H = opsum(FiniteChain(B, N), (pair(i, i + 1) for i in 1:(N - 1)), (0.5 * n[i] for i in 1:N))
    W = mpo(H)
    @test length(W) == N && W.lattice == H.lattice
    @test mpo_terms(W) ≈ H.terms
    _, op = first((Tuple(k), v) for (k, v) in OpSum.storedpairs(W[2]) if !isone(v))
    @test instantiate(op, B) isa AbstractTensorMap
    @test mpo_tensormap(irrep_mpo_tensors(W)) ≈ instantiate(H)

    # decaying channels on a finite chain: compressed, but no Jordan form yet
    Hd = opsum(FiniteChain(B, N), expterm(couple(n[1], n[2]); decay = 0.5))
    Wd = mpo(Hd)
    @test isempty(Wd.start)
    @test_throws ArgumentError jordan_mpo_tensors(Wd)
    @test_throws ArgumentError instantiate(Wd)
end

@testset "an infinite chain with a decaying interaction" begin
    # Σ_{i<j} λ^{j-i-1} nᵢ nⱼ + h Σ nᵢ against MPSKit's own construction, truncated at range R
    λ, hx, R = 0.4, 0.3, 40
    H = opsum(InfiniteChain(B), expterm(couple(n[1], n[2]); decay = λ), hx * n[1])
    Hmps = instantiate(mpo(H))
    nn = num ⊗ num
    ref = InfiniteMPOHamiltonian(
        [V], ((1, 1 + r) => λ^(r - 1) * nn for r in 1:R)..., (1,) => hx * num
    )
    ψ = InfiniteMPS([V], [Vmps])
    @test expectation_value(ψ, Hmps) ≈ expectation_value(ψ, ref) atol = 1.0e-10
    @test abs(expectation_value(ψ, ref)) > 1.0e-3
end
