using Test
using OpSum
using OpSum: Terms, scalarop
using TensorKit
using LinearAlgebra: dot, I as Id
using TensorKitTensors.HubbardOperators: HubbardOperators as HO
using TensorKitTensors.FermionOperators: FermionOperators as FO
using TensorKitTensors.SpinOperators: SpinOperators as SO

include(joinpath(@__DIR__, "..", "testutils.jl"))

# `a * b` across sites is `couple` with the forced channel; a product sharing a site is resolved in the
# site's basis when it enters an `OperatorSum`. The oracle is the dense operator product.

dense(lat, x) = plain(instantiate(opsum(lat, x)))

# One chain length per space type (three sites; four only where two pairs must interleave), so each compiles its
# contractions once; the fermionic site is the same space in the first and third testsets.

const Vf = Vect[FermionParity](0 => 1, 1 => 1)
const Bf = unitbasis(Vf)
const F = fermionmode(Bf, FermionParity(0), FermionParity(1))
const lat3 = FiniteChain(Bf, 3)

# the SU(2) Hubbard site, in a basis built from the spin-summed hopping pair
const P, Sy = U1Irrep, SU2Irrep
const Bh = Basis([HO.e_plus_e_min(ComplexF64, P, Sy), HO.e_min_e_plus(ComplexF64, P, Sy), HO.e_num(ComplexF64, P, Sy)])
const E = project(HO.e_plus_e_min(ComplexF64, P, Sy), Bh)
const nh = project(HO.e_num(ComplexF64, P, Sy), Bh)
const lath = FiniteChain(Bh, 4)

@testset "fermions: single-site physics and the dense product" begin
    lat1 = lat3
    one1 = dense(lat1, scalarop(1, Bf)[1])
    @test dense(lat1, F.cd[1] * F.c[1]) ≈ dense(lat1, F.n[1])            # c†c = n
    @test dense(lat1, F.c[1] * F.cd[1]) ≈ one1 - dense(lat1, F.n[1])     # cc† = 1 - n
    @test dense(lat1, F.n[1] * F.n[1]) ≈ dense(lat1, F.n[1])
    @test isempty(opsum(lat1, F.cd[1] * F.cd[1]))                        # c†c† = 0: no terms

    # across sites, `*` is `couple` with the forced channel, in either order
    @test opsum(lat3, F.cd[1] * F.c[3]).terms ≈ couple(F.cd[1], F.c[3])
    @test opsum(lat3, F.cd[3] * F.c[1]).terms ≈ couple(F.cd[3], F.c[1])
    @test (F.cd[1] * F.c[3]) ≈ couple(F.cd[1], F.c[3])                     # no shared site: resolved

    h12 = couple(F.cd[1], F.c[2]) + couple(F.cd[2], F.c[1])
    h23 = couple(F.cd[2], F.c[3]) + couple(F.cd[3], F.c[2])
    D(x) = dense(lat3, x)
    @test D(h12 * h23) ≈ D(h12) * D(h23)
    @test D(h12 * h12) ≈ D(h12) * D(h12)
    @test D(h12 * F.n[2] * h23) ≈ D(h12) * D(F.n[2]) * D(h23)
    # charged factors under an abelian symmetry: the channel is forced
    Dc(x) = instantiate(opsum(lat3, x))                                  # keeps the charge leg
    @test Dc(F.cd[1] * F.cd[2] * F.c[2]) ≈ Dc(couple(F.cd[1], F.n[2]; to = FermionParity(1)))
    @test D(h12 * (h12 + 1.5 * one(h12))) ≈ D(h12) * (D(h12) + 1.5 * D(scalarop(1, Bf)[1] + 0 * F.n[1]))
end

@testset "a product that shares a site has no normal form until it is placed in a sum" begin
    x = F.n[1] * F.n[1]
    @test_throws ArgumentError length(x)
    @test_throws ArgumentError x ≈ F.n[1]
    @test occursin("unresolved", sprint(show, x))
    @test_throws ArgumentError couple(x, F.c[2])
    @test_throws ArgumentError expterm(x * couple(F.cd[1], F.c[2]); decay = 0.5)
    # sums, scaling and opsum carry it along
    y = opsum(2 * x, F.n[2], x)
    @test dense(lat3, y) ≈ 3 * dense(lat3, F.n[1]) + dense(lat3, F.n[2])
    # an `OperatorSum` holds resolved terms only
    H = opsum(lat3, y)
    @test H.terms ≈ opsum(3 * F.n[1], F.n[2])
end

# a spinless fermion site in a basis built from the hopping pairs: 𝟙, n, c†, c
function fermion_site(P = Trivial)
    V = FO.fermion_space(P)
    hop, hop′ = FO.f_plus_f_min(ComplexF64, P), FO.f_min_f_plus(ComplexF64, P)
    num = FO.f_num(ComplexF64, P)
    B = Basis([hop, hop′, num])
    return (; V, B, h = project(hop, B), h′ = project(hop′, B), n = project(num, B))
end

@testset "a pair placed on one site is the on-site product" begin
    f = fermion_site()
    lat = FiniteChain(f.B, 3)
    D(x) = dense(lat, x)
    @test D(f.h[2, 2]) ≈ D(f.n[2])                             # Σ c†c on one site is n
    @test D(f.h′[2, 2]) ≈ D(scalarop(1, sectortype(f.V))[2] + 0 * f.n[2]) - D(f.n[2])
    # out of order on distinct sites (abelian): the same operator with the legs reordered
    @test D(f.h[2, 1]) ≈ -D(f.h′[1, 2])                        # c†₂c₁ = -c₁c†₂
    # and products with it, in the basis
    @test D(f.h[1, 2] * f.h[2, 3]) ≈ D(f.h[1, 2]) * D(f.h[2, 3])
    @test D(f.h[1, 1] * f.h[1, 2]) ≈ D(f.n[1]) * D(f.h[1, 2])
end

@testset "non-abelian: neutral factors, on one site and across" begin
    V = SU2Space(1 // 2 => 1)
    BU = unitbasis(V)
    S = spinvector(BU)
    SS = SO.S_exchange(ComplexF64, SU2Irrep)
    B = Basis([SS])
    ss = project(SS, B)
    lat = FiniteChain(B, 4)
    D(x) = dense(lat, x)
    @test D(ss[2, 2]) ≈ (3 / 4) * D(scalarop(1, SU2Irrep)[2] + 0 * ss[1, 2])   # S⃗·S⃗ = s(s+1)
    @test D(ss[1, 2] * ss[2, 3]) ≈ D(ss[1, 2]) * D(ss[2, 3])                     # shares site 2
    @test D(ss[1, 3] * ss[2, 3]) ≈ D(ss[1, 3]) * D(ss[2, 3])                     # interleaved, shared site
    @test (ss[3, 4] * ss[1, 2]) ≈ (ss[1, 2] * ss[3, 4])                           # disjoint: they commute
    @test D(ss[1, 3] * ss[2, 4]) ≈ D(ss[1, 3]) * D(ss[2, 4])                     # interleaved: re-coupled
    @test ss[2, 1] ≈ ss[1, 2]
    @test_throws ArgumentError S[1] * S[2]                        # charged: the channel is not forced
    # a charged factor first: the two caterpillars are merged (F-moves), then re-coupled
    Dc(x) = instantiate(opsum(lat, x))
    @test Dc(S[3] * ss[1, 2]) ≈ Dc(ss[1, 2] * S[3])

    # the SU(2) Hubbard site: Σσ a†σ aσ on one site is n
    @test dense(lath, E[1, 1]) ≈ dense(lath, nh[1])
    # a·n on one site is the second spinor element, which this basis lacks: resolving it throws
    @test_throws ArgumentError opsum(lath, E[1, 2] * E[2, 2])
end

# Re-coupling oracle: the written-order caterpillar materialised by `_written_block`, against the site-order terms.
function recouples(sites, keys, coeff, Vs)
    out = Terms{sectortype(first(Vs))}()
    OpSum._written!(out, sites, keys, ComplexF64(coeff))
    return OpSum.dense(out, Vs) ≈ coeff * OpSum._written_block(keys, sites, Vs)
end

@testset "non-abelian re-coupling against the written-order product" begin
    J = SU2Irrep
    B1 = unitbasis(SU2Space(1 // 2 => 1))
    S1 = spinvector(B1)
    letter(x) = only(only(x).keys).letter
    Vs = fill(B1, 4)
    for j in (0, 1, 2), tot in J(j) ⊗ J(1)                       # every channel, three sites
        a = only(couple(S1[1], S1[3]; to = J(j)))
        keys = vcat(a.keys, OpSum.Factor{J}(letter(S1[2]), tot))
        @test recouples([1, 3, 2], keys, 0.7, Vs)
        @test recouples([3, 2, 1], keys, 1.0, Vs)
    end
    a = only(couple(couple(S1[1], S1[3]; to = J(2)), S1[4]; to = J(1)))   # four sites, two orders
    keys = vcat(a.keys, OpSum.Factor{J}(letter(S1[2]), J(1)))
    @test recouples([1, 3, 4, 2], keys, 0.7, Vs)
    @test recouples([4, 1, 3, 2], keys, 1.0, Vs)
    # `couple` out of order gives the same terms
    a = only(couple(S1[1], S1[3]; to = J(1)))
    x = couple(Terms{J}([a]), S1[2]; to = J(1))
    keys = vcat(a.keys, OpSum.Factor{J}(letter(S1[2]), J(1)))
    @test OpSum.dense(x, Vs) ≈
        a.coeff * only(S1[2]).coeff * OpSum._written_block(keys, [1, 3, 2], Vs)

    # under an abelian symmetry the re-coupling is the R-symbol bookkeeping
    t = only(couple(F.cd[1], F.c[2], F.cd[3], F.c[4]))
    for p in ([3, 1, 4, 2], [2, 4, 1, 3], [4, 3, 2, 1])
        out = Terms{FermionParity}()
        OpSum._written!(out, p, t.keys, 1.0 + 0im)
        @test Terms{FermionParity}(OpSum._recouple(p, t.keys, 1.0 + 0im)) ≈ out
    end

    # fermions under SU(2): E[q, p] = E[p, q]†, and interleaved products of spin-summed hoppings
    @test dense(lath, E[3, 1]) ≈ dense(lath, E[1, 3])'
    D(x) = dense(lath, x)
    @test D(E[1, 3] * E[2, 4]) ≈ D(E[1, 3]) * D(E[2, 4])
end

@testset "on-site products on an infinite chain" begin
    f = fermion_site()
    # n_i² + hopping, one-site cell: resolved per insertion like any other term
    Hp = opsum(InfiniteChain(f.B), f.n[1] * f.n[1], -(f.h[1, 2] - f.h′[1, 2]))
    Hr = opsum(InfiniteChain(f.B), f.n[1], -(f.h[1, 2] - f.h′[1, 2]))
    @test Hp.terms ≈ Hr.terms
    @test mpo(Hp).bondsectors == mpo(Hr).bondsectors
end
