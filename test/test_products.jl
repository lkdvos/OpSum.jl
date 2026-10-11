using Test
using OpSum
using OpSum: Terms, scalarop
using TensorKit
using TensorKit: removeunit, numind
using LinearAlgebra: dot, I as Id
using TensorKitTensors.HubbardOperators: HubbardOperators as HO
using TensorKitTensors.FermionOperators: FermionOperators as FO

include("testutils.jl")

# Products of operators (operator-bases.md §3.1 item 7, §3.5): `a * b` across sites is `couple` with
# the forced channel, and a product that shares a site is resolved in the site's basis when it enters
# an `OperatorSum`. The oracle throughout is the dense operator product.

plain(t) = numind(t) == 2 * numout(t) ? t : removeunit(t, numind(t))
dense(lat, x) = plain(instantiate(opsum(lat, x)))
pairop(ts, V) = plain(instantiate(opsum(FiniteChain(V, 2), ts)))

const Vf = Vect[FermionNumber](0 => 1, 1 => 1)
const F = fermion_ops(Vf)

@testset "fermions in matrix units: single-site physics and the dense product" begin
    lat1, lat3 = FiniteChain(Vf, 1), FiniteChain(Vf, 3)
    one1 = dense(lat1, scalarop(1, Vf)[1])
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
    @test D(h23 * h12) ≈ D(h23) * D(h12)
    @test D(h12 * h12) ≈ D(h12) * D(h12)
    @test D(h12 * F.n[2] * h23) ≈ D(h12) * D(F.n[2]) * D(h23)
    # charged factors under an abelian symmetry: the channel is forced
    Dc(x) = instantiate(opsum(lat3, x))                                  # keeps the charge leg
    @test Dc(F.cd[1] * F.cd[2] * F.c[2]) ≈ Dc(couple(F.cd[1], F.n[2]; to = FermionNumber(1)))
    # scalars and the identity
    @test D((2 * one(h12)) * h12) ≈ 2 * D(h12)
    @test D(h12 * (h12 + 1.5 * one(h12))) ≈ D(h12) * (D(h12) + 1.5 * D(scalarop(1, Vf)[1] + 0 * F.n[1]))
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
    @test dense(FiniteChain(Vf, 2), y) ≈ 3 * dense(FiniteChain(Vf, 2), F.n[1]) + dense(FiniteChain(Vf, 2), F.n[2])
    # an `OperatorSum` holds resolved terms only
    H = opsum(FiniteChain(Vf, 2), y)
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
    @test D(f.h′[2, 2]) ≈ D(scalarop(1, Trivial ⊠ FermionParity)[2] + 0 * f.n[2]) - D(f.n[2])
    # out of order on distinct sites (abelian): the same operator with the legs reordered
    @test D(f.h[2, 1]) ≈ -D(f.h′[1, 2])                        # c†₂c₁ = -c₁c†₂
    @test D(f.h[3, 1]) ≈ -D(f.h′[1, 3])                        # across a gap
    # and products with it, in the basis
    @test D(f.h[1, 2] * f.h[2, 3]) ≈ D(f.h[1, 2]) * D(f.h[2, 3])
    @test D(f.h[1, 1] * f.h[1, 2]) ≈ D(f.n[1]) * D(f.h[1, 2])
end

@testset "non-abelian: neutral factors, on one site and across" begin
    V = SU2Space(1 // 2 => 1)
    S = spin(V)
    SS = pairop(dot(S[1], S[2]), V)
    B = Basis([SS])
    ss = project(SS, B)
    lat = FiniteChain(B, 4)
    D(x) = dense(lat, x)
    @test D(ss[2, 2]) ≈ (3 / 4) * D(scalarop(1, SU2Irrep)[2] + 0 * ss[1, 2])   # S⃗·S⃗ = s(s+1)
    @test D(ss[1, 2] * ss[2, 3]) ≈ D(ss[1, 2]) * D(ss[2, 3])                     # shares site 2
    @test D(ss[2, 3] * ss[1, 2]) ≈ D(ss[2, 3]) * D(ss[1, 2])
    @test D(ss[1, 3] * ss[2, 3]) ≈ D(ss[1, 3]) * D(ss[2, 3])                     # interleaved, shared site
    @test D(ss[3, 4] * ss[1, 2]) ≈ D(ss[3, 4]) * D(ss[1, 2])                     # disjoint: they commute
    @test (ss[3, 4] * ss[1, 2]) ≈ (ss[1, 2] * ss[3, 4])
    @test D(ss[1, 3] * ss[2, 4]) ≈ D(ss[1, 3]) * D(ss[2, 4])                     # interleaved: re-coupled
    @test D(ss[1, 4] * ss[3, 2]) ≈ D(ss[1, 4]) * D(ss[3, 2])
    @test ss[2, 1] ≈ ss[1, 2]
    @test_throws ArgumentError S[1] * S[2]                        # charged: the channel is not forced
    # a charged factor first: the two caterpillars are merged (F-moves), then re-coupled
    Dc(x) = instantiate(opsum(lat, x))
    @test Dc(S[3] * ss[1, 2]) ≈ Dc(ss[1, 2] * S[3])
    @test Dc(S[2] * ss[1, 3]) ≈ Dc(ss[1, 3] * S[2])

    # the SU(2) Hubbard site: Σσ a†σ aσ on one site is n
    P, Sy = U1Irrep, SU2Irrep
    Vh = HO.hubbard_space(P, Sy)
    Bh = Basis([HO.e_plus_e_min(ComplexF64, P, Sy), HO.e_min_e_plus(ComplexF64, P, Sy), HO.e_num(ComplexF64, P, Sy)])
    E = project(HO.e_plus_e_min(ComplexF64, P, Sy), Bh)
    n = project(HO.e_num(ComplexF64, P, Sy), Bh)
    lath = FiniteChain(Bh, 2)
    @test dense(lath, E[1, 1]) ≈ dense(lath, n[1])
    # a·n on one site is the second spinor element, which this basis lacks: resolving it throws
    @test_throws ArgumentError opsum(lath, E[1, 2] * E[2, 2])
end

# The oracle for re-coupling: the written-order caterpillar, materialised as written
# (`_written_block` multiplies the letters' tensors in written order and fuses their charge legs by the
# written tree), against the site-order terms it is re-expressed as.
function recouples(sites, keys, coeff, Vs)
    out = Terms{sectortype(first(Vs))}()
    OpSum._written!(out, sites, keys, ComplexF64(coeff))
    return OpSum._instantiate_terms(out, Vs) ≈ coeff * OpSum._written_block(keys, sites, Vs)
end

@testset "non-abelian re-coupling against the written-order product" begin
    J = SU2Irrep
    S1 = spin(SU2Space(1 => 1))
    letter(x) = only(only(x).keys).op
    Vs = fill(SU2Space(1 => 1), 4)
    for j in (0, 1, 2), tot in J(j) ⊗ J(1)                       # every channel, three sites
        a = only(couple(S1[1], S1[3]; to = J(j)))
        keys = vcat(a.keys, OpSum.ITOKey{J}(letter(S1[2]), tot, 1))
        @test recouples([1, 3, 2], keys, 0.7, Vs[1:3])
        @test recouples([3, 2, 1], keys, 1.0, Vs[1:3])
    end
    a = only(couple(couple(S1[1], S1[3]; to = J(2)), S1[4]; to = J(1)))   # four sites, two orders
    keys = vcat(a.keys, OpSum.ITOKey{J}(letter(S1[2]), J(1), 1))
    @test recouples([1, 3, 4, 2], keys, 0.7, Vs)
    @test recouples([4, 1, 3, 2], keys, 1.0, Vs)
    # `couple` out of order gives the same terms
    a = only(couple(S1[1], S1[3]; to = J(1)))
    x = couple(Terms{J}([a]), S1[2]; to = J(1))
    keys = vcat(a.keys, OpSum.ITOKey{J}(letter(S1[2]), J(1), 1))
    @test OpSum._instantiate_terms(x, Vs[1:3]) ≈
        a.coeff * only(S1[2]).coeff * OpSum._written_block(keys, [1, 3, 2], Vs[1:3])

    # under an abelian symmetry the re-coupling is the R-symbol bookkeeping
    t = only(couple(F.cd[1], F.c[2], F.cd[3], F.c[4]))
    for p in ([3, 1, 4, 2], [2, 4, 1, 3], [4, 3, 2, 1])
        out = Terms{FermionNumber}()
        OpSum._written!(out, p, t.keys, 1.0 + 0im)
        @test Terms{FermionNumber}(OpSum._recouple(p, t.keys, 1.0 + 0im)) ≈ out
    end

    # fermions under SU(2): E[q, p] = E[p, q]†, and interleaved products of spin-summed hoppings
    P, Sy = U1Irrep, SU2Irrep
    Bh = Basis([HO.e_plus_e_min(ComplexF64, P, Sy), HO.e_min_e_plus(ComplexF64, P, Sy), HO.e_num(ComplexF64, P, Sy)])
    E = project(HO.e_plus_e_min(ComplexF64, P, Sy), Bh)
    lat3, lat4 = FiniteChain(Bh, 3), FiniteChain(Bh, 4)
    @test dense(lat3, E[3, 1]) ≈ dense(lat3, E[1, 3])'
    @test dense(lat3, E[2, 1]) ≈ dense(lat3, E[1, 2])'
    D(x) = dense(lat4, x)
    @test D(E[1, 3] * E[2, 4]) ≈ D(E[1, 3]) * D(E[2, 4])
    @test D(E[4, 1] * E[3, 2]) ≈ D(E[4, 1]) * D(E[3, 2])
end

@testset "on-site products on an infinite chain" begin
    f = fermion_site()
    # n_i² + hopping, one-site cell: resolved per insertion like any other term
    Hp = opsum(InfiniteChain(f.B), f.n[1] * f.n[1], -(f.h[1, 2] - f.h′[1, 2]))
    Hr = opsum(InfiniteChain(f.B), f.n[1], -(f.h[1, 2] - f.h′[1, 2]))
    @test Hp.terms ≈ Hr.terms
    @test mpo(Hp).bondsectors == mpo(Hr).bondsectors
end
