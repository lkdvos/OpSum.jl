using Test
using OpSum
using OpSum: instantiate, Term, total, passthrough, ispassthrough, bondcharges, scalarop, couple, opsum, tree
using OpSum: Letter
using TensorKit
using LinearAlgebra: dot, I as Id

include(joinpath(@__DIR__, "..", "testutils.jl"))   # LO, onlyterm

charges(t::Term) = [k.op.c for k in t.keys]
termbonds(t::Term) = [k.bond for k in t.keys]
treebonds(t::Term) = bondcharges(tree(t))      # the keys *are* the tree

@testset "pass-through identity symbol" begin
    pt = passthrough(SU2Irrep)
    @test ispassthrough(pt) && pt.c == unit(SU2Irrep) && pt.n == 1
    V = SU2Space(1 // 2 => 1)
    @test all(!ispassthrough, instances(Letter, V))
    @test reshape(convert(Array, instantiate(pt, V)), 2, 2) ≈ Matrix{ComplexF64}(Id, 2, 2)
end

@testset "SU(2) Sᵢ·Sⱼ term normal form" begin
    S = spinvector(unitbasis(SU2Space(1 // 2 => 1)))
    ts = dot(S[1], S[2])
    @test length(ts) == 1
    tk = onlyterm(ts)
    @test tk.sites == [1, 2]
    @test charges(tk) == [SU2Irrep(1), SU2Irrep(1)]
    @test total(tk) == unit(SU2Irrep)
    @test termbonds(tk) == [SU2Irrep(1), SU2Irrep(0)] == treebonds(tk)
    @test tk.coeff ≈ (3 / 2) * (-sqrt(3))               # ⟨½‖S‖½⟩² · (−√dim(1))

    tk2 = onlyterm(couple(S[1], S[2]; to = SU2Irrep(1)))   # non-singlet: no −√dim factor
    @test tk2.coeff ≈ 3 / 2
    @test termbonds(tk2) == [SU2Irrep(1), SU2Irrep(1)] == treebonds(tk2)
end

@testset "U(1) hopping term normal form" begin
    raise = LO(Letter(U1Irrep(1), 1))
    lower = LO(Letter(U1Irrep(-1), 1))
    tk = onlyterm(dot(raise[1], lower[2]))
    @test charges(tk) == [U1Irrep(1), U1Irrep(-1)]
    @test total(tk) == unit(U1Irrep)
    @test termbonds(tk) == [U1Irrep(1), U1Irrep(0)] == treebonds(tk)
    @test tk.coeff ≈ -1.0
    @test termbonds(onlyterm(couple(raise[1], raise[2]; to = U1Irrep(2)))) == [U1Irrep(1), U1Irrep(2)]

    # charges that cannot reach the unit sector have no scalar product, not even on the same site
    @test_throws ArgumentError dot(raise[1], raise[2])
    @test_throws ArgumentError dot(raise[1], raise[1])
end

@testset "scalar field is a K = 0 term" begin
    tk0 = onlyterm(scalarop(2.0, ℂ^2)[1])
    @test isempty(tk0.sites) && tk0.coeff ≈ 2.0 && total(tk0) == unit(Trivial)
end

@testset "K=3 SU(2): distinct coupling channels" begin
    V = SU2Space(1 // 2 => 1)
    S = spinvector(unitbasis(V))
    terms = [couple(couple(S[1], S[2]; to = SU2Irrep(b)), S[3]; to = SU2Irrep(1)) for b in 0:2]
    ks = onlyterm.(terms)
    @test length(unique(ks)) == 3
    @test all(k -> k.sites == [1, 2, 3] && total(k) == SU2Irrep(1), ks)
    @test [termbonds(k) for k in ks] == [[SU2Irrep(1), SU2Irrep(b), SU2Irrep(1)] for b in 0:2]
    mats = [convert(Array, instantiate(opsum(fill(unitbasis(V), 3), t))) for t in terms]
    @test !any(mats[i] ≈ mats[j] for i in 1:3 for j in (i + 1):3)
end

# round-off left over from a cancellation is no difference, a term of comparable size is; `==` is exact
@testset "≈ on term sets tolerates round-off leftovers" begin
    S = spinvector(unitbasis(SU2Space(1 // 2 => 1)))
    a, b = dot(S[1], S[2]), dot(S[2], S[3])
    @test a + 1.0e-17 * b ≈ a
    @test a + 1.0e-17 * b != a
    @test !(a + 1.0e-3 * b ≈ a)
    @test !(a ≈ a + b) && !(a + b ≈ a)
    @test isapprox(a, a + 1.0e-3 * b; atol = 1.0e-2)
end
