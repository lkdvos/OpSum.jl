using Test
using OpSum
using OpSum: instantiate, total, bondcharges
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using LinearAlgebra: dot, norm, eigvals, I as Id

include(joinpath(@__DIR__, "testutils.jl"))   # LO

# dense reference Pauli / spin-1/2
const σX = ComplexF64[0 1; 1 0]
const σY = ComplexF64[0 -im; im 0]
const σZ = ComplexF64[1 0; 0 -1]

@testset "OnsiteOp scalar / sum instantiation" begin
    V = SU2Space(1 // 2 => 1)

    # scalar identity → structural c·id(V), NOT via one(::Type)
    Osc = instantiate(scalarop(2.0, V), V)
    @test space(Osc) == (V ← V)
    @test convert(Array, Osc) ≈ 2 * Matrix{ComplexF64}(Id, 2, 2)

    # scalarop can also be built from the sector type
    @test space(instantiate(scalarop(1, SU2Irrep), V)) == (V ← V)

    # bare ITO
    Osp = instantiate(spin(V), V)
    @test space(Osp) == (V ← V ⊗ Vect[SU2Irrep](SU2Irrep(1) => 1))
    @test norm(convert(Array, Osp)) > 0

    # sum (scaled) distributes over instantiate
    S2 = spin(V) + 0.5 * spin(V)
    @test convert(Array, instantiate(S2, V)) ≈ 1.5 * convert(Array, Osp)

    # spin normalization: reduced element √(s(s+1)(2s+1)) = √(3/2) for spin-1/2
    coeff = only(values(spin(V)))
    @test coeff ≈ sqrt(3 / 2)

    # On-site products are not symbolic (see research/onsite-products.md). The supported route is to
    # build the product as a `TensorMap` and project it; `n̂² = n̂` checks it against a known answer.
    Vu = Rep[U₁](0 => 1, 1 => 1)
    nhat = matrixunit(Vu, U1Irrep(1), U1Irrep(1))
    Nop = zeros(ComplexF64, Vu ← Vu)
    block(Nop, U1Irrep(1)) .= 1
    @test project(Nop * Nop, Vu) ≈ nhat
end

@testset "single-site field embedding" begin
    V = SU2Space(1 // 2 => 1)
    B = convert(Array, instantiate(spin(V), V))          # (2,2,3)
    Wf = convert(Array, instantiate(spin(V)[2], [V, V, V]))
    @test size(Wf) == (2, 2, 2, 2, 2, 2, 3)
    I2 = Matrix{ComplexF64}(Id, 2, 2)
    oracle = [
        I2[o1, i1] * B[o2, i2, q] * I2[o3, i3]
            for o1 in 1:2, o2 in 1:2, o3 in 1:2, i1 in 1:2, i2 in 1:2, i3 in 1:2, q in 1:3
    ]
    @test Wf ≈ oracle

    # empty-site identity is the structural identity on all sites
    idop = one(spin(V)[1])
    @test reshape(convert(Array, instantiate(idop, [V, V])), 4, 4) ≈
        Matrix{Float64}(Id, 4, 4)
end

@testset "SU(2) two-site S·S = Cartesian dot" begin
    # spin-1/2 : ¼(σx⊗σx + σy⊗σy + σz⊗σz)
    V = SU2Space(1 // 2 => 1)
    Sx, Sy, Sz = σX / 2, σY / 2, σZ / 2
    Wd = dropdims(convert(Array, instantiate(dot(spin(V)[1], spin(V)[2]), [V, V])); dims = 5)
    orc = [
        Sx[o1, i1] * Sx[o2, i2] + Sy[o1, i1] * Sy[o2, i2] + Sz[o1, i1] * Sz[o2, i2]
            for o1 in 1:2, o2 in 1:2, i1 in 1:2, i2 in 1:2
    ]
    @test Wd ≈ orc
    # explicit ¼σ·σ form
    quarter = 0.25 * (kron(σX, σX) + kron(σY, σY) + kron(σZ, σZ))
    @test reshape(permutedims(Wd, (2, 1, 4, 3)), 4, 4) ≈ quarter

    # spin-1 Heisenberg
    V1 = SU2Space(1 => 1)
    sq2 = sqrt(2.0)
    Sz1 = ComplexF64[1 0 0; 0 0 0; 0 0 -1]
    Sp1 = ComplexF64[0 sq2 0; 0 0 sq2; 0 0 0]
    Sm1 = collect(Sp1')
    Sx1, Sy1 = (Sp1 + Sm1) / 2, (Sp1 - Sm1) / (2im)
    Wd1 = dropdims(convert(Array, instantiate(dot(spin(V1)[1], spin(V1)[2]), [V1, V1])); dims = 5)
    orc1 = [
        Sx1[o1, i1] * Sx1[o2, i2] + Sy1[o1, i1] * Sy1[o2, i2] + Sz1[o1, i1] * Sz1[o2, i2]
            for o1 in 1:3, o2 in 1:3, i1 in 1:3, i2 in 1:3
    ]
    @test Wd1 ≈ orc1
end

@testset "trivial-sector coupling (ℂ²)" begin
    V = ℂ^2
    ops = instances(IrrepOperator, V)
    E2 = convert(Array, instantiate(ops[2], V))[:, :, 1]
    E3 = convert(Array, instantiate(ops[3], V))[:, :, 1]
    # bare `couple` carries no factor (the Cartesian −√dim lives in `·`/`dot`, not here)
    Cd = dropdims(convert(Array, instantiate(couple(LO(ops[2])[1], LO(ops[3])[2]; to = unit(Trivial)), [V, V])); dims = 5)
    orc = [E2[o1, i1] * E3[o2, i2] for o1 in 1:2, o2 in 1:2, i1 in 1:2, i2 in 1:2]
    @test Cd ≈ orc
end

@testset "U(1) coupling / charge structure" begin
    V = Rep[U₁](0 => 1, 1 => 1)
    raise = LO(IrrepOperator(U1Irrep(1), 1))
    lower = LO(IrrepOperator(U1Irrep(-1), 1))

    # singlet hopping term is number-conserving (net charge 0)
    Hop = instantiate(dot(raise[1], lower[2]), [V, V])
    @test space(Hop) == ((V ⊗ V) ← (V ⊗ V ⊗ Vect[U1Irrep](U1Irrep(0) => 1)))
    rd = dropdims(convert(Array, instantiate(raise, V)); dims = 3)
    ld = dropdims(convert(Array, instantiate(lower, V)); dims = 3)
    Hd = dropdims(convert(Array, Hop); dims = 5)
    orc = [-rd[o1, i1] * ld[o2, i2] for o1 in 1:2, o2 in 1:2, i1 in 1:2, i2 in 1:2]
    @test Hd ≈ orc

    # coupling two charge +1 operators to a definite total charge +2
    C2 = instantiate(couple(raise[1], raise[2]; to = U1Irrep(2)), [V, V])
    @test collect(blocksectors(C2)) == [U1Irrep(2)]
end

@testset "coupling API errors / deferrals" begin
    V = SU2Space(1 // 2 => 1)
    a, b, c = spin(V)[1], spin(V)[2], spin(V)[3]
    # tree coupling deferred to Phase 3
    @test_throws ArgumentError couple(a, b; to = SU2Irrep(0), via = :tree)
    @test_throws ArgumentError couple(a, b, c; to = SU2Irrep(0), via = :tree)
    # variadic coupling needs unique fusion; SU(2) intermediates are a real choice
    @test_throws ArgumentError couple(a, b, c; to = SU2Irrep(0))
    @test_throws ArgumentError couple(a, b, c)
    # coupling a scalar (no charge leg) is invalid
    @test_throws ArgumentError instantiate(couple(scalarop(1, V)[1], b; to = SU2Irrep(0)), [V, V, V])
    # same-site coupling invalid
    @test_throws ArgumentError instantiate(couple(spin(V)[1], spin(V)[1]; to = SU2Irrep(0)), [V, V])
end

@testset "`to` defaults to the unit sector" begin
    Vu = Rep[U₁](0 => 1, 1 => 1)
    dn, up = U1Irrep(0), U1Irrep(1)
    Sp, Sm = matrixunit(Vu, up, dn), matrixunit(Vu, dn, up)
    Sz = (matrixunit(Vu, up, up) - matrixunit(Vu, dn, dn)) / 2

    @test couple(Sp[1], Sm[2]).terms == couple(Sp[1], Sm[2]; to = unit(U1Irrep)).terms
    @test couple(Sz[1], Sz[2]).terms == couple(Sz[1], Sz[2]; to = unit(U1Irrep)).terms

    # charges that cannot reach the unit sector are an error, not a silent empty result
    @test_throws ArgumentError couple(Sp[1], Sp[2])
    @test length(couple(Sp[1], Sp[2]; to = U1Irrep(2)).terms) == 1

    # non-abelian and fermionic defaults too
    V2 = SU2Space(1 // 2 => 1)
    @test total(only(keys(couple(spin(V2)[1], spin(V2)[2]).terms))) == unit(SU2Irrep)
    Vf = Vect[FermionNumber](0 => 1, 1 => 1)
    cc = matrixunit(Vf, FermionNumber(0), FermionNumber(1))
    cd = matrixunit(Vf, FermionNumber(1), FermionNumber(0))
    @test total(only(keys(couple(cd[1], cc[2]).terms))) == unit(FermionNumber)
end

@testset "variadic coupling (abelian)" begin
    Vu = Rep[U₁](0 => 1, 1 => 1)
    dn, up = U1Irrep(0), U1Irrep(1)
    z = unit(U1Irrep)
    Sp, Sm = matrixunit(Vu, up, dn), matrixunit(Vu, dn, up)
    Sz = (matrixunit(Vu, up, up) - matrixunit(Vu, dn, dn)) / 2

    # the intermediates are forced, so the variadic form must agree with the explicit nesting
    @test couple(Sp[1], Sm[2], Sz[3]).terms ==
        couple(couple(Sp[1], Sm[2]; to = z), Sz[3]; to = z).terms

    Vf = Vect[FermionNumber](0 => 1, 1 => 1)
    u = unit(FermionNumber)
    cc = matrixunit(Vf, FermionNumber(0), FermionNumber(1))
    cd = matrixunit(Vf, FermionNumber(1), FermionNumber(0))
    H4 = couple(cd[1], cc[2], cd[3], cc[4])
    nested = couple(
        couple(couple(cd[1], cc[2]; to = u), cd[3]; to = FermionNumber(1)), cc[4]; to = u
    )
    @test H4.terms == nested.terms
    @test only(keys(H4.terms)).sites == [1, 2, 3, 4]
    @test bondcharges(only(keys(H4.terms)).tree) ==
        [FermionNumber(1), u, FermionNumber(1), u]

    # composite operands still distribute over every chain
    @test length(couple(Sz[1], Sz[2], Sz[3]).terms) == 8

    # a total the charges cannot reach is an error
    @test_throws ArgumentError couple(cd[1], cd[2], cd[3], cc[4])
    @test length(couple(Sp[1], Sp[2], Sm[3]; to = U1Irrep(1)).terms) == 1

    # independent dense check: S⁺₁ S⁻₂ Sᶻ₃, with the sector-ordered basis (0 => dn, 1 => up)
    Spm = ComplexF64[0 0; 1 0]
    Smm = ComplexF64[0 1; 0 0]
    Szm = ComplexF64[-0.5 0; 0 0.5]
    A = dropdims(convert(Array, instantiate(couple(Sp[1], Sm[2], Sz[3]), fill(Vu, 3))); dims = 7)
    ref = [
        Spm[o1, i1] * Smm[o2, i2] * Szm[o3, i3]
            for o1 in 1:2, o2 in 1:2, o3 in 1:2, i1 in 1:2, i2 in 1:2, i3 in 1:2
    ]
    @test A ≈ ref
end

# ── Out-of-order coupling ─────────────────────────────────────────────────────
# The stored form is site-ordered, so an operand acting to the left of an earlier one has its leg
# *inserted* and the coefficient picks up the braiding phase. That is exactly the fermionic
# anticommutation sign the caller used to have to supply, so these tests pin it against the site-
# ordered spelling *and*, where possible, against an independent oracle (hermiticity, and the analytic
# free-fermion spectrum).

# spectrum computed block by block, so this is valid for fermionic sectors (`convert(Array, ·)` is not)
function blockspectrum(H, sites)
    O = instantiate(H, sites)
    Oop = numind(O) == 2 * numout(O) ? O : removeunit(O, numind(O))
    vals = Float64[]
    for (c, b) in blocks(Oop)
        ev = real(eigvals(Matrix(b)))
        for _ in 1:dim(c)
            append!(vals, ev)
        end
    end
    return sort!(vals)
end

function relative_hermiticity_error(H, sites)
    O = instantiate(H, sites)
    Oop = numind(O) == 2 * numout(O) ? O : removeunit(O, numind(O))
    return norm(Oop - Oop') / norm(Oop)
end

@testset "out-of-order couple inserts the braiding phase" begin
    Vf = Vect[FermionNumber](0 => 1, 1 => 1)
    F = fermion_ops(Vf)

    # two odd legs swapped: R = -1
    @test couple(F.cd[2], F.c[1]) ≈ -couple(F.c[1], F.cd[2])
    @test couple(F.c[2], F.cd[1]) ≈ -couple(F.cd[1], F.c[2])
    # an even leg (n̂ carries the unit charge) swaps for free
    @test couple(F.n[2], F.n[1]) ≈ couple(F.n[1], F.n[2])
    @test couple(F.n[2], F.cd[1]; to = FermionNumber(1)) ≈
        couple(F.cd[1], F.n[2]; to = FermionNumber(1))

    # bosonic abelian sectors: no phase at all
    Vu = Rep[U₁](0 => 1, 1 => 1)
    S = spin_ops(Vu, U1Irrep(1), U1Irrep(0))
    @test couple(S.Sm[2], S.Sp[1]) ≈ couple(S.Sp[1], S.Sm[2])
    @test couple(S.Sz[3], S.Sz[1]) ≈ couple(S.Sz[1], S.Sz[3])

    # the inserted leg lands in site order with the running bond charges recomputed
    t = only(keys(couple(F.cd[2], F.c[1]).terms))
    @test t.sites == [1, 2]
    @test bondcharges(t.tree) == [FermionNumber(-1), unit(FermionNumber)]

    # insertion in the *middle* of an existing caterpillar
    lhs = couple(couple(F.cd[1], F.c[4]), F.n[2])
    rhs = couple(couple(F.cd[1], F.n[2]; to = FermionNumber(1)), F.c[4])
    @test lhs ≈ rhs
    @test only(keys(lhs.terms)).sites == [1, 2, 4]

    # a four-leg permutation: reordering [3,1,4,2] has three inversions among odd legs, so -1
    @test couple(F.cd[3], F.c[1], F.cd[4], F.c[2]) ≈
        -couple(F.c[1], F.c[2], F.cd[3], F.cd[4])

    # same site is still an error, whichever order it is written in
    @test_throws ArgumentError couple(F.cd[2], F.c[2])
    @test_throws ArgumentError couple(couple(F.cd[1], F.c[3]), F.n[3])

    # non-abelian reordering needs F-moves and is refused
    Vs = SU2Space(1 // 2 => 1)
    S2 = spin(Vs)
    @test_throws ArgumentError couple(couple(S2[1], S2[3]; to = SU2Irrep(1)), S2[2])
end

@testset "the sign out-of-order couple supplies is the physical one" begin
    Vf = Vect[FermionNumber](0 => 1, 1 => 1)
    F = fermion_ops(Vf)
    N, t = 6, 1.0
    sites = fill(Vf, N)

    # the textbook spelling of a hopping term, hermitian and with the analytic spectrum
    H = -t * sum([couple(F.cd[i], F.c[i + 1]) + couple(F.cd[i + 1], F.c[i]) for i in 1:(N - 1)])
    @test relative_hermiticity_error(H, sites) < 1.0e-12
    ε = [-2t * cos(k * π / (N + 1)) for k in 1:N]
    exact = sort(
        [sum(ε[k] for k in 1:N if (m >> (k - 1)) & 1 == 1; init = 0.0) for m in 0:(2^N - 1)]
    )
    @test blockspectrum(H, sites) ≈ exact
    # the hand-signed spelling it replaces
    @test H ≈ -t * sum([couple(F.cd[i], F.c[i + 1]) - couple(F.c[i], F.cd[i + 1]) for i in 1:(N - 1)])
end

@testset "dot accepts either site order, with the swap phase" begin
    # `dot` couples two legs to the unit sector, which needs no F-move whatever the fusion style, so
    # it works out of order where the general `couple` reordering does not.
    Vs = SU2Space(1 // 2 => 1)
    S = spin(Vs)
    @test dot(S[3], S[1]) ≈ dot(S[1], S[3])          # integer operator charge: R = +1

    Vu = Rep[U₁](0 => 1, 1 => 1)
    Su = spin_ops(Vu, U1Irrep(1), U1Irrep(0))
    @test dot(Su.Sm[3], Su.Sp[1]) ≈ dot(Su.Sp[1], Su.Sm[3])

    # odd fermionic charges anticommute, so `dot` is *not* order-independent there — the previous
    # implementation sorted the sites and dropped this sign
    Vf = Vect[FermionNumber](0 => 1, 1 => 1)
    F = fermion_ops(Vf)
    @test dot(F.cd[3], F.c[1]) ≈ -dot(F.c[1], F.cd[3])

    # half-integer SU(2) operator charges pick it up too (R(½, ½, 0) = -1)
    Vh = SU2Space(0 => 1, 1 // 2 => 1)
    spinor = LO(IrrepOperator(SU2Irrep(1 // 2), 1))
    @test dot(spinor[3], spinor[1]) ≈ -dot(spinor[1], spinor[3])

    @test_throws ArgumentError dot(S[1], S[1])
end

@testset "hc is the hermitian conjugate" begin
    Vf = Vect[FermionNumber](0 => 1, 1 => 1)
    F = fermion_ops(Vf)
    N = 4
    sites = fill(Vf, N)

    T = onlattice(-1.0 * sum([couple(F.cd[i], F.c[i + 1]) for i in 1:(N - 1)]), sites)
    @test T + hc(T) ≈
        -1.0 * sum([couple(F.cd[i], F.c[i + 1]) + couple(F.cd[i + 1], F.c[i]) for i in 1:(N - 1)])
    @test relative_hermiticity_error(T + hc(T), sites) < 1.0e-12
    @test hc(hc(T)) ≈ T
    @test lattice(hc(T)) == sites
    # `sites` may be given instead of bound
    @test hc(-1.0 * couple(F.cd[1], F.c[2]), sites) ≈ hc(onlattice(-1.0 * couple(F.cd[1], F.c[2]), sites))

    # a complex amplitude has to be conjugated, and a longer-range hop crosses a site
    for T2 in (
            (0.3 + 0.7im) * couple(F.cd[1], F.c[2]),
            0.5im * couple(F.cd[1], F.c[3]),
            (1.0 + 0.5im) * couple(F.cd[1], F.cd[2], F.c[3], F.c[4]),
            2.0im * F.n[2] + scalarop(1.0 + 1.0im, Vf)[1],
        )
        Tl = onlattice(T2, sites)
        @test relative_hermiticity_error(Tl + hc(Tl), sites) < 1.0e-12
        @test hc(hc(Tl)) ≈ Tl
    end

    # non-abelian, including a 3-body term with a genuine inner line
    Vs = SU2Space(1 // 2 => 1)
    S = spin(Vs)
    Hs = onlattice(sum([dot(S[i], S[i + 1]) for i in 1:3]), fill(Vs, 4))
    @test hc(Hs) ≈ Hs
    T3 = onlattice(
        (0.4 + 0.2im) * couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0)),
        fill(Vs, 3)
    )
    @test relative_hermiticity_error(T3 + hc(T3), fill(Vs, 3)) < 1.0e-12
    @test hc(hc(T3)) ≈ T3

    # a charged term's adjoint lives in the dual sector, so it is refused
    @test_throws ArgumentError hc(onlattice(F.cd[1], sites))
    @test isempty(hc(onlattice(TermSum{FermionNumber}(), sites)))
end

@testset "operator builders" begin
    # U(1) spin ladders, against the SU(2) build of the same Heisenberg chain. Two different symmetry
    # groups and two different spaces: agreeing spectra is the sharpest available cross-check, and it
    # pins the ladder normalisation for s > 1/2 as well.
    for (su2, u1secs) in (
            (SU2Space(1 // 2 => 1), U1Irrep[U1Irrep(1), U1Irrep(0)]),
            (SU2Space(1 => 1), U1Irrep[U1Irrep(1), U1Irrep(0), U1Irrep(-1)]),
            (SU2Space(3 // 2 => 1), U1Irrep[U1Irrep(k) for k in 3:-1:0]),
        )
        L = 4
        S = spin(su2)
        Hsu2 = sum([dot(S[i], S[i + 1]) for i in 1:(L - 1)])
        Vu = Vect[U1Irrep](c => 1 for c in u1secs)
        F = spin_ops(Vu, u1secs)
        Hu1 = sum(
            [
                couple(F.Sz[i], F.Sz[i + 1]) +
                    (couple(F.Sp[i], F.Sm[i + 1]) + couple(F.Sm[i], F.Sp[i + 1])) / 2
                    for i in 1:(L - 1)
            ]
        )
        @test blockspectrum(Hsu2, fill(su2, L)) ≈ blockspectrum(Hu1, fill(Vu, L))
        @test islossless(Hu1, fill(Vu, L))
    end

    # the two-sector form is the block it replaces
    V2 = Rep[U₁](0 => 1, 1 => 1)
    up, dn = U1Irrep(1), U1Irrep(0)
    S = spin_ops(V2, up, dn)
    @test S == spin_ops(V2, U1Irrep[up, dn])
    @test S.Sp ≈ matrixunit(V2, up, dn)
    @test S.Sm ≈ matrixunit(V2, dn, up)
    @test S.Sz ≈ (matrixunit(V2, up, up) - matrixunit(V2, dn, dn)) / 2
    @test_throws ArgumentError spin_ops(V2, U1Irrep[up])
    @test_throws ArgumentError spin_ops(V2, U1Irrep[up, up])
    # a *partial* sector list is the trap the length-derived `s` makes possible: it would build
    # spin-½ ladder operators on two of three sectors and never say so
    V3 = Rep[U₁](-1 => 1, 0 => 1, 1 => 1)
    @test_throws ArgumentError spin_ops(V3, U1Irrep(1), U1Irrep(0))
    @test spin_ops(V3, U1Irrep[U1Irrep(1), U1Irrep(0), U1Irrep(-1)]).Sz ≈
        matrixunit(V3, U1Irrep(1), U1Irrep(1)) - matrixunit(V3, U1Irrep(-1), U1Irrep(-1))

    Vf = Vect[FermionNumber](0 => 1, 1 => 1)
    vac, occ = FermionNumber(0), FermionNumber(1)
    F = fermion_ops()
    @test F == fermion_ops(Vf, vac, occ)
    @test F.c ≈ matrixunit(Vf, vac, occ)
    @test F.cd ≈ matrixunit(Vf, occ, vac)
    @test F.n ≈ matrixunit(Vf, occ, occ)
    # only `FermionNumber` names its own vacuum
    @test_throws ArgumentError fermion_ops(V2)

    # memoised, so building inside a term loop is a lookup
    @test spin(SU2Space(1 // 2 => 1)) === spin(SU2Space(1 // 2 => 1))
    @test matrixunit(Vf, vac, occ) === matrixunit(Vf, vac, occ)
end
