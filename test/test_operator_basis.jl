using Test
using OpSum
using OpSum: SiteOperator, LocalOperator, nsites, passthrough, ispassthrough, scalarop, Terms,
    _instantiate_terms, _space
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using TensorKit: removeunit, numind
using LinearAlgebra: dot

include("testutils.jl")

const Imat = ComplexF64[1 0; 0 1]
const Xmat = ComplexF64[0 1; 1 0]
const Ymat = ComplexF64[0 -im; im 0]
const Zmat = ComplexF64[1 0; 0 -1]
const Spmat = ComplexF64[0 1; 0 0]
const Smmat = ComplexF64[0 0; 1 0]

# a `V ← V` operator from the `V ← V ⊗ unit` form `instantiate` gives
plain(t) = numind(t) == 2 * numout(t) ? t : removeunit(t, numind(t))

# the dense operator of a placed term bag over per-site bases (or spaces)
dense(ts, sites) = plain(_instantiate_terms(ts, sites))

# a neutral two-site operator, built through the matrix-unit pipeline
pairop(ts, V) = plain(instantiate(opsum(FiniteChain(V, 2), ts)))

# the dense bond dimension at bond `b`
bonddim(H, b) = ((_, secs) = irrep_mpo(H); densedim(secs, b))

@testset "Basis: identity first, letters per charge" begin
    B = Basis([Imat, Xmat, Ymat, Zmat])
    @test length(B) == 4                         # the explicit identity is the prepended one
    @test space(B) == ℂ^2
    @test B.letters == [IrrepOperator(Trivial(), n) for n in 1:4]
    @test first(B.letters) == passthrough(Trivial)
    @test plain(B[1]) ≈ id(ℂ^2)
    @test instantiate(passthrough(Trivial), B) ≈ B[1]

    # without the identity given, it is still element 1
    B2 = Basis([Zmat, Xmat])
    @test length(B2) == 3
    @test plain(B2[1]) ≈ id(ℂ^2)
    @test plain(B2[2]) ≈ TensorMap(Zmat, ℂ^2 ← ℂ^2)

    @test_throws ArgumentError Basis([Xmat, Ymat, Xmat + Ymat])   # dependent
    @test_throws ArgumentError Basis([Xmat, 2Imat + Zmat, Imat - Zmat])
    @test length(Basis([Xmat, 2Xmat])) == 2                        # parallel: deduplicated
    @test_throws ArgumentError Basis(Any[])
end

@testset "project(O, basis) on one space" begin
    B = Basis([Imat, Xmat, Ymat, Zmat])
    X = project(Xmat, B)
    @test collect(pairs(X)) == [IrrepOperator(Trivial(), 2) => 1]
    Sp = project(Spmat, B)                                   # (X + iY)/2: two elements
    @test Sp ≈ 0.5 * IrrepOperator(Trivial(), 2) + 0.5im * IrrepOperator(Trivial(), 3)
    @test plain(instantiate(Sp, B)) ≈ TensorMap(Spmat, ℂ^2 ← ℂ^2)

    # the identity is element 1, i.e. a scalar
    @test project(Imat, B) == scalarop(1, Trivial)
    @test project(2Imat + Zmat, B) ≈ scalarop(2, Trivial) + project(Zmat, B)
    @test isempty(project(zeros(2, 2), B))

    # the same operators, atomic in another (non-orthogonal) basis
    A = Basis([Imat, Spmat, Smmat, Zmat + Imat])
    @test length(A) == 4
    @test collect(pairs(project(Spmat, A))) == [IrrepOperator(Trivial(), 2) => 1]
    Z = project(Zmat, A)
    @test Z ≈ -1 * passthrough(Trivial) + 1 * IrrepOperator(Trivial(), 4)
    @test plain(instantiate(project(Xmat, A), A)) ≈ TensorMap(Xmat, ℂ^2 ← ℂ^2)

    # outside the span
    @test_throws ArgumentError project(Xmat, Basis([Zmat]))
    @test_throws ArgumentError project(ones(3, 3), B)
    @test_throws ArgumentError project(TensorMap(Xmat, ℂ^2 ← ℂ^2), Basis([Xmat[1:1, 1:1]]))
end

@testset "a projected identity slot places as a pass-through" begin
    B = Basis([Xmat, Zmat])
    XI = project(TensorMap(kron(Imat, Xmat), ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2), B)   # X on slot 1
    @test XI isa LocalOperator && nsites(XI) == 2
    @test XI[3] ≈ project(Xmat, B)[3]                 # drops the identity slot
    @test all(t -> length(t.sites) == 1, XI[3])
end

@testset "XX+YY+ZZ chain: the sweep runs on basis indices" begin
    N = 8
    V = ℂ^2
    P = Basis([Xmat, Ymat, Zmat])
    Xp, Yp, Zp = (project(M, P) for M in (Xmat, Ymat, Zmat))
    Hp = opsum(FiniteChain(V, N), (couple(O[i], O[i + 1]) for i in 1:(N - 1) for O in (Xp, Yp, Zp)))
    @test islossless(Hp)
    @test bonddim(Hp, N ÷ 2) == 5                      # handoff §5: 6 today, 5 in the Pauli basis

    # the matrix-unit alphabet on the same model
    Xm, Ym, Zm = (project(TensorMap(M, V ← V), V) for M in (Xmat, Ymat, Zmat))
    Hm = opsum(FiniteChain(V, N), (couple(O[i], O[i + 1]) for i in 1:(N - 1) for O in (Xm, Ym, Zm)))
    @test bonddim(Hm, N ÷ 2) == 6

    # both are the same operator: the basis MPO, read back through the basis, against the dense one
    n = 4
    Hp4 = opsum(FiniteChain(V, n), (couple(O[i], O[i + 1]) for i in 1:(n - 1) for O in (Xp, Yp, Zp)))
    Hm4 = opsum(FiniteChain(V, n), (couple(O[i], O[i + 1]) for i in 1:(n - 1) for O in (Xm, Ym, Zm)))
    @test dense(mpo_terms(irrep_mpo(Hp4)), fill(P, n)) ≈ plain(instantiate(Hm4))
end

@testset "U(1) spins: charged elements by factorising neutral pairs" begin
    V = Rep[U₁](0 => 1, 1 => 1)
    L = spin_ops(V, U1Irrep(1), U1Irrep(0))
    SpSm = pairop(couple(L.Sp[1], L.Sm[2]), V)
    SmSp = pairop(couple(L.Sm[1], L.Sp[2]), V)
    Sz = plain(instantiate(opsum(FiniteChain(V, 1), L.Sz[1])))
    B = Basis([SpSm, SmSp, Sz])
    @test length(B) == 4                                    # 𝟙, Sᶻ, S⁺, S⁻: the pieces are shared
    @test Set(keys(B.bycharge)) == Set(U1Irrep[-1, 0, 1])
    @test Basis([SpSm, SmSp, Sz]).elements == B.elements   # deterministic

    # a rank-one pair is its two pieces, with coefficient one
    pm = project(SpSm, B)
    @test length(pm.terms) == 1 && only(pm.terms).coeff ≈ 1
    @test dense(pm[1], fill(B, 2)) ≈ SpSm
    z = project(Sz, B)
    @test length(z) == 1

    # XXZ chain, with a next-nearest-neighbour term across a gap
    N = 6
    mp = project(SmSp, B)
    H = opsum(
        FiniteChain(V, N),
        ((pm[i] + mp[i]) / 2 + 0.7 * couple(z[i], z[i + 1]) for i in 1:(N - 1)),
        (0.3 * pm[i, i + 2] for i in 1:(N - 2)),
    )
    @test islossless(H)
    Hm = opsum(
        FiniteChain(V, N),
        (
            (couple(L.Sp[i], L.Sm[i + 1]) + couple(L.Sm[i], L.Sp[i + 1])) / 2 +
                0.7 * couple(L.Sz[i], L.Sz[i + 1]) for i in 1:(N - 1)
        ),
        (0.3 * couple(L.Sp[i], L.Sm[i + 2]) for i in 1:(N - 2)),
    )
    @test dense(mpo_terms(irrep_mpo(H)), fill(B, N)) ≈ plain(instantiate(Hm))
    @test bonddim(H, N ÷ 2) <= bonddim(Hm, N ÷ 2)
end

@testset "fermions: the identity slot carries the sign across a gap" begin
    V = Vect[FermionNumber](0 => 1, 1 => 1)
    F = fermion_ops(V)
    hop = pairop(couple(F.cd[1], F.c[2]), V)
    hop′ = pairop(couple(F.c[1], F.cd[2]), V)
    n = plain(instantiate(opsum(FiniteChain(V, 1), F.n[1])))
    B = Basis([hop, hop′, n])
    @test length(B) == 4                                    # 𝟙, n, c†, c
    h = project(hop, B)
    @test length(h.terms) == 1

    # c†₁ 𝟙₂ c₃ projected as a block equals the pair placed across the gap
    h3 = plain(instantiate(opsum(FiniteChain(V, 3), couple(F.cd[1], F.c[3]))))
    b3 = project(h3, B)
    @test b3[1] ≈ h[1, 3]
    @test dense(b3[1], fill(B, 3)) ≈ h3

    # placement is monotone for now, so c†ᵢ₊₁cᵢ = -cᵢc†ᵢ₊₁ is the other pair, on the same pieces
    h′ = project(hop′, B)
    N = 5
    H = opsum(
        FiniteChain(V, N),
        (-(h[i, i + 1] - h′[i, i + 1]) for i in 1:(N - 1)),
        (0.4 * h[i, i + 2] for i in 1:(N - 2)),
        (0.5 * project(n, B)[i] for i in 1:N),
    )
    @test islossless(H)
    Hm = opsum(
        FiniteChain(V, N),
        (-(couple(F.cd[i], F.c[i + 1]) + couple(F.cd[i + 1], F.c[i])) for i in 1:(N - 1)),
        (0.4 * couple(F.cd[i], F.c[i + 2]) for i in 1:(N - 2)),
        (0.5 * F.n[i] for i in 1:N),
    )
    @test dense(mpo_terms(irrep_mpo(H)), fill(B, N)) ≈ plain(instantiate(Hm))
end

@testset "SU(2): the exchange pair factorises into one spin-1 element" begin
    V = SU2Space(1 // 2 => 1)
    S = spin(V)
    SS = pairop(dot(S[1], S[2]), V)
    B = Basis([SS])
    @test length(B) == 2                                    # 𝟙 and S⃗ (left and right piece agree)
    # the right piece is minus the left one (the singlet's coupling sign), and the left one is kept,
    # so the gauge shows up as the pair's coefficient
    ss = project(SS, B)
    @test length(ss.terms) == 1 && only(ss.terms).coeff ≈ -1
    N = 4
    H = opsum(FiniteChain(V, N), (ss[i] for i in 1:(N - 1)))
    @test islossless(H)
    Hm = opsum(FiniteChain(V, N), (dot(S[i], S[i + 1]) for i in 1:(N - 1)))
    @test dense(mpo_terms(irrep_mpo(H)), fill(B, N)) ≈ plain(instantiate(Hm))
    # a constant part is an all-identity term, which places as a scalar
    sc = project(SS + 2 * id(V ⊗ V), B)
    @test sc[1] ≈ ss[1] + 2 * Terms{SU2Irrep}(OpSum.Term{SU2Irrep}[OpSum.Term(Int[], OpSum.ITOKey{SU2Irrep}[], 1)])
end
