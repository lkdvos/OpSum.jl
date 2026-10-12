using Test
using OpSum
using OpSum: instantiate, total, bondcharges, tree, opsum
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using LinearAlgebra: dot, norm, eigvals, I as Id

include(joinpath(@__DIR__, "testutils.jl"))

# dense oracle of a latticeless bag over a list of sites; a bare space stands for its `unitbasis`
sitebasis(V::ElementarySpace) = unitbasis(V)
sitebasis(B::Basis) = B
inst(ts, sites) = instantiate(opsum(map(sitebasis, sites), ts))

# U(1) spin-½ ladder operators in the stock basis, where `Sz` is a single letter
const U1B = spin_basis(U1Irrep)
const U1UP, U1DN = U1Irrep(1 // 2), U1Irrep(-1 // 2)
u1ladder() = spinladder(U1B, [U1UP, U1DN])

# dense reference Pauli / spin-1/2
const σX = ComplexF64[0 1; 1 0]
const σY = ComplexF64[0 -im; im 0]
const σZ = ComplexF64[1 0; 0 -1]

@testset "SiteOperator scalar / sum instantiation" begin
    V = SU2Space(1 // 2 => 1)

    # scalar identity → c·𝟙 with a trivial charge leg, like every other letter
    Vu = Vect[SU2Irrep](SU2Irrep(0) => 1)
    Osc = instantiate(scalarop(2.0, V), V)
    @test space(Osc) == (V ← V ⊗ Vu)
    @test reshape(convert(Array, Osc), 2, 2) ≈ 2 * Matrix{ComplexF64}(Id, 2, 2)

    @test space(instantiate(scalarop(1, SU2Irrep), V)) == (V ← V ⊗ Vu)

    B = unitbasis(V)
    S = spinvector(B)
    Osp = instantiate(S, B)
    @test space(Osp) == (V ← V ⊗ Vect[SU2Irrep](SU2Irrep(1) => 1))
    @test norm(convert(Array, Osp)) > 0

    # sum (scaled) distributes over instantiate
    S2 = S + 0.5 * S
    @test convert(Array, instantiate(S2, B)) ≈ 1.5 * convert(Array, Osp)

    # spin normalization: reduced element √(s(s+1)(2s+1)) = √(3/2) for spin-1/2
    coeff = only(values(S))
    @test coeff ≈ sqrt(3 / 2)

    # On-site products are not symbolic: the supported route is to build the product as a
    # `TensorMap` and project it. `n̂² = n̂` checks that route against a known answer.
    Vu = Rep[U₁](0 => 1, 1 => 1)
    Bu = unitbasis(Vu)
    nhat = matrixunit(Bu, U1Irrep(1), U1Irrep(1))
    Nop = zeros(ComplexF64, Vu ← Vu)
    block(Nop, U1Irrep(1)) .= 1
    @test project(Nop * Nop, Bu) ≈ nhat
end

@testset "single-site field embedding" begin
    V = SU2Space(1 // 2 => 1)
    S = spinvector(unitbasis(V))
    B = convert(Array, instantiate(S, unitbasis(V)))     # (2,2,3)
    Wf = convert(Array, inst(S[2], [V, V, V]))
    @test size(Wf) == (2, 2, 2, 2, 2, 2, 3)
    I2 = Matrix{ComplexF64}(Id, 2, 2)
    oracle = [
        I2[o1, i1] * B[o2, i2, q] * I2[o3, i3]
            for o1 in 1:2, o2 in 1:2, o3 in 1:2, i1 in 1:2, i2 in 1:2, i3 in 1:2, q in 1:3
    ]
    @test Wf ≈ oracle

    # empty-site identity is the structural identity on all sites
    idop = one(S[1])
    @test reshape(convert(Array, inst(idop, [V, V])), 4, 4) ≈
        Matrix{Float64}(Id, 4, 4)
end

@testset "SU(2) two-site S·S = Cartesian dot" begin
    # spin-1/2 : ¼(σx⊗σx + σy⊗σy + σz⊗σz)
    V = SU2Space(1 // 2 => 1)
    Sx, Sy, Sz = σX / 2, σY / 2, σZ / 2
    S = spinvector(unitbasis(V))
    Wd = dropdims(convert(Array, inst(dot(S[1], S[2]), [V, V])); dims = 5)
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
    S1 = spinvector(unitbasis(V1))
    Wd1 = dropdims(convert(Array, inst(dot(S1[1], S1[2]), [V1, V1])); dims = 5)
    orc1 = [
        Sx1[o1, i1] * Sx1[o2, i2] + Sy1[o1, i1] * Sy1[o2, i2] + Sz1[o1, i1] * Sz1[o2, i2]
            for o1 in 1:3, o2 in 1:3, i1 in 1:3, i2 in 1:3
    ]
    @test Wd1 ≈ orc1
end

@testset "trivial-sector coupling (ℂ²)" begin
    V = ℂ^2
    B = unitbasis(V)
    ops = [l for l in B.letters if !OpSum.ispassthrough(l)]    # the two trivial-charge letters of ℂ²: here, all of them
    E2 = convert(Array, instantiate(LO(ops[1]), B))[:, :, 1]
    E3 = convert(Array, instantiate(LO(ops[2]), B))[:, :, 1]
    # bare `couple` carries no factor (the Cartesian −√dim lives in `·`/`dot`, not here)
    Cd = dropdims(convert(Array, inst(couple(LO(ops[1])[1], LO(ops[2])[2]; to = unit(Trivial)), [V, V])); dims = 5)
    orc = [E2[o1, i1] * E3[o2, i2] for o1 in 1:2, o2 in 1:2, i1 in 1:2, i2 in 1:2]
    @test Cd ≈ orc
end

@testset "U(1) coupling / charge structure" begin
    V = Rep[U₁](0 => 1, 1 => 1)
    raise = LO(IrrepOperator(U1Irrep(1), 1))
    lower = LO(IrrepOperator(U1Irrep(-1), 1))

    # singlet hopping term is number-conserving (net charge 0)
    Hop = inst(dot(raise[1], lower[2]), [V, V])
    @test space(Hop) == ((V ⊗ V) ← (V ⊗ V ⊗ Vect[U1Irrep](U1Irrep(0) => 1)))
    rd = dropdims(convert(Array, instantiate(raise, V)); dims = 3)
    ld = dropdims(convert(Array, instantiate(lower, V)); dims = 3)
    Hd = dropdims(convert(Array, Hop); dims = 5)
    orc = [-rd[o1, i1] * ld[o2, i2] for o1 in 1:2, o2 in 1:2, i1 in 1:2, i2 in 1:2]
    @test Hd ≈ orc

    # coupling two charge +1 operators to a definite total charge +2
    C2 = inst(couple(raise[1], raise[2]; to = U1Irrep(2)), [V, V])
    @test collect(blocksectors(C2)) == [U1Irrep(2)]
end

@testset "coupling API errors / deferrals" begin
    V = SU2Space(1 // 2 => 1)
    S = spinvector(unitbasis(V))
    a, b, c = S[1], S[2], S[3]
    @test_throws ArgumentError couple(a, b; to = SU2Irrep(0), via = :tree)
    @test_throws ArgumentError couple(a, b, c; to = SU2Irrep(0), via = :tree)
    # four rank-1 operators leave (j₁₂, j₁₂₃) free, and the error says so rather than picking one
    d = S[4]
    err = try
        couple(a, b, c, d)
    catch e
        e isa ArgumentError || rethrow()
        sprint(showerror, e)
    end
    @test err isa String && occursin("genuine choice", err) && occursin("couple_channels", err)
    @test_throws ArgumentError inst(couple(scalarop(1, V)[1], b; to = SU2Irrep(0)), [V, V, V])
    @test_throws ArgumentError inst(couple(S[1], S[1]; to = SU2Irrep(0)), [V, V])
end

@testset "`to` defaults to the unit sector" begin
    (; Sp, Sm, Sz) = u1ladder()

    @test collect(couple(Sp[1], Sm[2])) == collect(couple(Sp[1], Sm[2]; to = unit(U1Irrep)))
    @test collect(couple(Sz[1], Sz[2])) == collect(couple(Sz[1], Sz[2]; to = unit(U1Irrep)))

    # charges that cannot reach the unit sector are an error, not a silent empty result
    @test_throws ArgumentError couple(Sp[1], Sp[2])
    @test length(couple(Sp[1], Sp[2]; to = U1Irrep(2))) == 1

    # non-abelian and fermionic defaults too
    V2 = SU2Space(1 // 2 => 1)
    S2 = spinvector(unitbasis(V2))
    @test total(only(couple(S2[1], S2[2]))) == unit(SU2Irrep)
    Vf = Vect[FermionNumber](0 => 1, 1 => 1)
    (; c, cd) = fermionmode(unitbasis(Vf), FermionNumber(0), FermionNumber(1))
    @test total(only(couple(cd[1], c[2]))) == unit(FermionNumber)
end

@testset "forced channels, non-abelian" begin
    V = SU2Space(1 // 2 => 1)
    S = spinvector(unitbasis(V))
    zero_, one_ = SU2Irrep(0), SU2Irrep(1)

    # three rank-1 operators reach a singlet only through j₁₂ = 1, so the fold is unambiguous
    @test couple_channels(S[1], S[2], S[3]; to = zero_) == [(one_,)]
    @test couple(S[1], S[2], S[3]) ≈ couple(couple(S[1], S[2]; to = one_), S[3]; to = zero_)
    @test couple(S[1], S[2], S[3]; to = zero_) ≈ couple(S[1], S[2], S[3])
    @test bondcharges(tree(only(couple(S[1], S[2], S[3])))) == [one_, one_, zero_]

    # four leave two of the three (j₁₂, j₁₂₃) tuples genuinely free
    @test couple_channels(S[1], S[2], S[3], S[4]) ==
        [(SU2Irrep(0), one_), (one_, one_), (SU2Irrep(2), one_)]
    @test_throws ArgumentError couple(S[1], S[2], S[3], S[4])
    # every tuple the query returns is a real operator, and nesting builds it
    for (j12, j123) in couple_channels(S[1], S[2], S[3], S[4])
        t = couple(couple(couple(S[1], S[2]; to = j12), S[3]; to = j123), S[4])
        @test length(t) == 1
        @test bondcharges(tree(only(t))) == [one_, j12, j123, zero_]
    end

    # two operands: nothing to name, so the query reports one empty tuple, or none at all
    @test couple_channels(S[1], S[2]) == [()]
    @test isempty(couple_channels(S[1], S[2]; to = SU2Irrep(5)))
    @test_throws ArgumentError couple(S[1], S[2], S[3]; to = SU2Irrep(5))
    @test_throws ArgumentError couple_channels(S[1])

    # the query agrees with the abelian case, where every channel is forced by construction
    Vf = Vect[FermionNumber](0 => 1, 1 => 1)
    (; c, cd) = fermionmode(unitbasis(Vf), FermionNumber(0), FermionNumber(1))
    @test couple_channels(cd[1], c[2], cd[3], c[4]) ==
        [(unit(FermionNumber), FermionNumber(1))]
end

@testset "variadic coupling (abelian)" begin
    z = unit(U1Irrep)
    (; Sp, Sm, Sz) = u1ladder()

    # the intermediates are forced, so the variadic form must agree with the explicit nesting
    @test collect(couple(Sp[1], Sm[2], Sz[3])) ==
        collect(couple(couple(Sp[1], Sm[2]; to = z), Sz[3]; to = z))

    Vf = Vect[FermionNumber](0 => 1, 1 => 1)
    u = unit(FermionNumber)
    (; c, cd) = fermionmode(unitbasis(Vf), FermionNumber(0), FermionNumber(1))
    H4 = couple(cd[1], c[2], cd[3], c[4])
    nested = couple(
        couple(couple(cd[1], c[2]; to = u), cd[3]; to = FermionNumber(1)), c[4]; to = u
    )
    @test collect(H4) == collect(nested)
    @test only(H4).sites == [1, 2, 3, 4]
    @test bondcharges(tree(only(H4))) ==
        [FermionNumber(1), u, FermionNumber(1), u]

    # composite operands still distribute over every chain: two neutral letters per site, 2³ chains
    # (a spin-1 basis, where P₀ + P₊ is a genuine two-letter operand)
    B3 = unitbasis(Rep[U₁](-1 => 1, 0 => 1, 1 => 1))
    Z = sum(
        element(B3, n) for n in eachindex(B3.letters)
            if B3.letters[n].c == unit(U1Irrep) && !OpSum.ispassthrough(B3.letters[n])
    )
    @test length(Z) == 2
    @test length(couple(Z[1], Z[2], Z[3])) == 8

    # a total the charges cannot reach is an error
    @test_throws ArgumentError couple(cd[1], cd[2], cd[3], c[4])
    @test length(couple(Sp[1], Sp[2], Sm[3]; to = U1Irrep(1))) == 1

    # independent dense check: S⁺₁ S⁻₂ Sᶻ₃, with the basis order of `spin_space(U1Irrep)` (up, dn)
    Spm = ComplexF64[0 1; 0 0]
    Smm = ComplexF64[0 0; 1 0]
    Szm = ComplexF64[0.5 0; 0 -0.5]
    A = dropdims(convert(Array, inst(couple(Sp[1], Sm[2], Sz[3]), fill(U1B, 3))); dims = 7)
    ref = [
        Spm[o1, i1] * Smm[o2, i2] * Szm[o3, i3]
            for o1 in 1:2, o2 in 1:2, o3 in 1:2, i1 in 1:2, i2 in 1:2, i3 in 1:2
    ]
    @test A ≈ ref
end

# Out-of-order coupling: an operand left of an earlier one has its leg inserted and the coefficient
# picks up the braiding phase; pinned against the site-ordered spelling and independent oracles.

# Spectrum computed block by block, so it is valid for any sector (`convert(Array, t)` is not, for fermionic ones).
function blockspectrum(H, sites)
    O = inst(H, sites)
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
    O = inst(H, sites)
    Oop = numind(O) == 2 * numout(O) ? O : removeunit(O, numind(O))
    return norm(Oop - Oop') / norm(Oop)
end

@testset "out-of-order couple inserts the braiding phase" begin
    Vf = Vect[FermionNumber](0 => 1, 1 => 1)
    F = fermionmode(unitbasis(Vf), FermionNumber(0), FermionNumber(1))

    # two odd legs swapped: R = -1
    @test couple(F.cd[2], F.c[1]) ≈ -couple(F.c[1], F.cd[2])
    @test couple(F.c[2], F.cd[1]) ≈ -couple(F.cd[1], F.c[2])
    # an even leg (n̂ carries the unit charge) swaps for free
    @test couple(F.n[2], F.n[1]) ≈ couple(F.n[1], F.n[2])
    @test couple(F.n[2], F.cd[1]; to = FermionNumber(1)) ≈
        couple(F.cd[1], F.n[2]; to = FermionNumber(1))

    # bosonic abelian sectors: no phase at all
    S = u1ladder()
    @test couple(S.Sm[2], S.Sp[1]) ≈ couple(S.Sp[1], S.Sm[2])
    @test couple(S.Sz[3], S.Sz[1]) ≈ couple(S.Sz[1], S.Sz[3])

    # the inserted leg lands in site order with the running bond charges recomputed
    t = only(couple(F.cd[2], F.c[1]))
    @test t.sites == [1, 2]
    @test bondcharges(tree(t)) == [FermionNumber(-1), unit(FermionNumber)]

    # insertion in the *middle* of an existing caterpillar
    lhs = couple(couple(F.cd[1], F.c[4]), F.n[2])
    rhs = couple(couple(F.cd[1], F.n[2]; to = FermionNumber(1)), F.c[4])
    @test lhs ≈ rhs
    @test only(lhs).sites == [1, 2, 4]

    # a four-leg permutation: reordering [3,1,4,2] has three inversions among odd legs, so -1
    @test couple(F.cd[3], F.c[1], F.cd[4], F.c[2]) ≈
        -couple(F.c[1], F.c[2], F.cd[3], F.cd[4])

    # same site is still an error, whichever order it is written in
    @test_throws ArgumentError couple(F.cd[2], F.c[2])
    @test_throws ArgumentError couple(couple(F.cd[1], F.c[3]), F.n[3])

    # non-abelian: re-coupled into site order (test_products.jl checks it against the dense product);
    # to a singlet, three spin-½ vectors leave only j₁₂ = 1
    Vs = SU2Space(1 // 2 => 1)
    S2 = spinvector(unitbasis(Vs))
    t = only(couple(couple(S2[1], S2[3]; to = SU2Irrep(1)), S2[2]))
    @test t.sites == [1, 2, 3]
    @test bondcharges(tree(t)) == SU2Irrep[1, 1, 0]
end

@testset "the sign out-of-order couple supplies is the physical one" begin
    Vf = Vect[FermionNumber](0 => 1, 1 => 1)
    Bf = unitbasis(Vf)
    F = fermionmode(Bf, FermionNumber(0), FermionNumber(1))
    N, t = 6, 1.0
    sites = fill(Bf, N)

    # the textbook spelling of a hopping term, hermitian and with the analytic spectrum
    H = opsum(
        (-t * (couple(F.cd[i], F.c[i + 1]) + couple(F.cd[i + 1], F.c[i])) for i in 1:(N - 1))
    )
    @test relative_hermiticity_error(H, sites) < 1.0e-12
    ε = [-2t * cos(k * π / (N + 1)) for k in 1:N]
    exact = sort(
        [sum(ε[k] for k in 1:N if (m >> (k - 1)) & 1 == 1; init = 0.0) for m in 0:(2^N - 1)]
    )
    @test blockspectrum(H, sites) ≈ exact
    @test H ≈ opsum(
        (-t * (couple(F.cd[i], F.c[i + 1]) - couple(F.c[i], F.cd[i + 1])) for i in 1:(N - 1))
    )
end

@testset "dot accepts either site order, with the swap phase" begin
    Vs = SU2Space(1 // 2 => 1)
    S = spinvector(unitbasis(Vs))
    @test dot(S[3], S[1]) ≈ dot(S[1], S[3])          # integer operator charge: R = +1

    Su = u1ladder()
    @test dot(Su.Sm[3], Su.Sp[1]) ≈ dot(Su.Sp[1], Su.Sm[3])

    # odd fermionic charges anticommute, so `dot` is not order-independent there
    Vf = Vect[FermionNumber](0 => 1, 1 => 1)
    F = fermionmode(unitbasis(Vf), FermionNumber(0), FermionNumber(1))
    @test dot(F.cd[3], F.c[1]) ≈ -dot(F.c[1], F.cd[3])

    # half-integer SU(2) operator charges pick it up too (R(½, ½, 0) = -1)
    spinor = LO(IrrepOperator(SU2Irrep(1 // 2), 1))
    @test dot(spinor[3], spinor[1]) ≈ -dot(spinor[1], spinor[3])

    @test_throws ArgumentError dot(S[1], S[1])
end

@testset "adjoint is the hermitian conjugate" begin
    Vf = Vect[FermionNumber](0 => 1, 1 => 1)
    Bf = unitbasis(Vf)
    F = fermionmode(Bf, FermionNumber(0), FermionNumber(1))
    N = 4
    sites = fill(Bf, N)

    T = opsum(-1.0 * couple(F.cd[i], F.c[i + 1]) for i in 1:(N - 1))
    Td = opsum(sites, T)'
    @test T + Td.terms ≈ opsum(
        -1.0 * (couple(F.cd[i], F.c[i + 1]) + couple(F.cd[i + 1], F.c[i])) for i in 1:(N - 1)
    )
    @test relative_hermiticity_error(T + Td.terms, sites) < 1.0e-12
    @test Td'.terms ≈ T

    # a complex amplitude has to be conjugated, and a longer-range hop crosses a site
    for T2 in (
            (0.3 + 0.7im) * couple(F.cd[1], F.c[2]),
            0.5im * couple(F.cd[1], F.c[3]),
            (1.0 + 0.5im) * couple(F.cd[1], F.cd[2], F.c[3], F.c[4]),
            2.0im * F.n[2] + scalarop(1.0 + 1.0im, Vf)[1],
        )
        Tl = opsum(T2)
        Tld = opsum(sites, Tl)'
        @test relative_hermiticity_error(Tl + Tld.terms, sites) < 1.0e-12
        @test Tld'.terms ≈ Tl
    end

    # non-abelian, including a 3-body term with a genuine inner line
    Vs = SU2Space(1 // 2 => 1)
    S = spinvector(unitbasis(Vs))
    ssites = fill(unitbasis(Vs), 4)
    Hs = opsum(dot(S[i], S[i + 1]) for i in 1:3)
    @test opsum(ssites, Hs)'.terms ≈ Hs
    T3 = opsum(
        (0.4 + 0.2im) * couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0))
    )
    T3d = opsum(ssites, T3)'
    @test relative_hermiticity_error(T3 + T3d.terms, ssites) < 1.0e-12
    @test T3d'.terms ≈ T3

    # a charged term's adjoint lives in the dual sector, so it is refused
    @test_throws ArgumentError opsum(sites, F.cd[1])'
    @test isempty(opsum(sites, Terms{FermionNumber}())')
    # and the postfix form says what it is missing rather than raising a MethodError
    @test_throws ArgumentError couple(F.cd[1], F.c[2])'
end

@testset "operator builders" begin
    # U(1) spin ladders against the SU(2) build of the same Heisenberg chain (pins the ladder normalisation)
    for s in (1 // 2, 1, 3 // 2)
        L = 4
        su2 = SU2Space(s => 1)
        S = spinvector(unitbasis(su2))
        Hsu2 = opsum((dot(S[i], S[i + 1]) for i in 1:(L - 1)))
        Bu = spin_basis(U1Irrep; spin = s)
        F = spinladder(Bu, [U1Irrep(m) for m in s:-1:(-s)])
        Hu1 = opsum(
            (
                F.Sz[i] * F.Sz[i + 1] +      # `*`: for s = 3/2 `Sz` carries an identity component, which `couple` refuses
                    (couple(F.Sp[i], F.Sm[i + 1]) + couple(F.Sm[i], F.Sp[i + 1])) / 2
                    for i in 1:(L - 1)
            )
        )
        @test blockspectrum(Hsu2, fill(unitbasis(su2), L)) ≈ blockspectrum(Hu1, fill(Bu, L))
        @test islossless(opsum(fill(Bu, L), Hu1))
    end
end
