using Test
using TensorKit
using LinearAlgebra: dot
using OpSum
using OpSum: mpo, mpo_tensors, mpo_terms_window, window_terms, unitcell_terms,
    translate, maxspan, termspan, couple, scalarop, instantiate
using OpSum: Letter

include(joinpath(@__DIR__, "testutils.jl"))   # unitbasis, element, matrixunit, spinvector

# Reference models: `(name, generating term bag, unit-cell spaces)`; the operator is `Σ_n translate(H, n·L)`.

const VSU2 = SU2Space(1 // 2 => 1)
const VU1 = U1Space(-1 // 2 => 1, 1 // 2 => 1)
const VTR = ComplexSpace(2)
const BSU2 = unitbasis(VSU2)
const BU1 = spin_basis(U1Irrep)   # stock basis: Sᶻ is one atomic letter
const BTR = unitbasis(VTR)

function su2_models()
    S = spinvector(BSU2)
    return [
        ("SU2 Heisenberg L=1", dot(S[1], S[2]), [BSU2]),
        ("SU2 Heisenberg L=2", dot(S[1], S[2]) + dot(S[2], S[3]), [BSU2, BSU2]),
        ("SU2 Heisenberg L=3", dot(S[1], S[2]) + dot(S[2], S[3]) + dot(S[3], S[4]), [BSU2, BSU2, BSU2]),
        ("SU2 dimerised L=2", dot(S[1], S[2]) + 0.4 * dot(S[2], S[3]), [BSU2, BSU2]),
        ("SU2 J1-J2 L=1", dot(S[1], S[2]) + 0.5 * dot(S[1], S[3]), [BSU2]),
        (
            "SU2 J1-J2 L=2", dot(S[1], S[2]) + dot(S[2], S[3]) +
                0.5 * (dot(S[1], S[3]) + dot(S[2], S[4])), [BSU2, BSU2],
        ),
        ("SU2 J1-J2-J3 L=1", dot(S[1], S[2]) + 0.5 * dot(S[1], S[3]) + 0.25 * dot(S[1], S[4]), [BSU2]),
        # K = 3: the caterpillar inner line is not fixed by (charges, total)
        ("SU2 3-site chain L=1", couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]), [BSU2]),
    ]
end

function u1_models()
    Sp = matrixunit(BU1, U1Irrep(1 // 2), U1Irrep(-1 // 2))
    Sm = matrixunit(BU1, U1Irrep(-1 // 2), U1Irrep(1 // 2))
    Sz = spin_ops(U1Irrep).S_z
    xxz(i, j; Δ = 1.0) =
        0.5 * couple(Sp[i], Sm[j]) + 0.5 * couple(Sm[i], Sp[j]) + Δ * couple(Sz[i], Sz[j])
    return [
        ("U1 XXZ L=1", xxz(1, 2; Δ = 0.7), [BU1]),
        # charge-0 on-site letter against a two-site tail: the pending/started suffix-class collision fires at every bond
        ("U1 XXZ + hSz L=1", xxz(1, 2) + 0.3 * Sz[1], [BU1]),
        ("U1 XXZ + hSz L=2", xxz(1, 2) + xxz(2, 3) + 0.3 * Sz[1] + 0.3 * Sz[2], [BU1, BU1]),
        ("U1 XXZ + staggered hSz L=2", xxz(1, 2) + xxz(2, 3) + 0.3 * Sz[1] - 0.3 * Sz[2], [BU1, BU1]),
        ("U1 next-nearest hop L=1", couple(Sp[1], Sm[3]) + couple(Sm[1], Sp[3]), [BU1]),
    ]
end

LO(x) = OpSum.SiteOperator(x)

# `ℂ^2` has multiplicity 2 in the trivial sector, so `matrixunit` does not apply; use `element`
function trivial_models()
    A, B = element(BTR, 2), element(BTR, 3)
    return [
        ("trivial 2-site L=1", couple(A[1], B[2]), [BTR]),
        # same collision
        ("trivial 2-site + on-site L=1", couple(A[1], B[2]) + 0.7 * B[1], [BTR]),
        ("trivial 3rd-neighbour L=1", couple(A[1], B[2]) + 0.5 * couple(A[1], B[4]), [BTR]),
    ]
end

reference_models() = vcat(su2_models(), u1_models(), trivial_models())

densedim(sec) = sum(dim, sec; init = 0)

# `mpo_terms_window` is a sandwich against `window_terms`: terms near the right edge of the window may be
# missing, everything produced must be correct and everything comfortably inside must be present.
function faithful(H, spaces; ncells = 5)
    L = length(spaces)
    H = opsum(InfiniteChain(spaces), H)
    gen = unitcell_terms(H)
    R = maxspan(gen)
    N = ncells * L
    N > R + 1 || error("window too small for the guard")
    # keyed by support (sites and factors), the coefficient is the value
    got = termdict(mpo_terms_window(mpo(H), ncells))
    want = termdict(window_terms(gen, ncells))

    for (t, v) in got
        haskey(want, t) || return false, "spurious term on sites $(first(t))"
        want[t] ≈ v || return false, "wrong coefficient on sites $(first(t))"
    end
    for (t, _) in want
        maximum(first(t)) <= N - R - 1 || continue
        haskey(got, t) || return false, "missing term on sites $(first(t))"
    end
    return true, ""
end

@testset "infinite MPO round-trips its generating term set" begin
    for (name, H, spaces) in reference_models()
        ok, why = faithful(H, spaces)
        @test ok || (println("  $name: $why"); false)
    end
end

@testset "the unit cell closes and reports two identity channels" begin
    for (name, H, spaces) in reference_models()
        L = length(spaces)
        Hinf = mpo(opsum(InfiniteChain(spaces), H))
        @test length(Hinf) == L
        for j in 1:L
            @test size(Hinf.Ws[j], 1) == length(Hinf.bondsectors[mod1(j - 1, L)])
            @test size(Hinf.Ws[j], 2) == length(Hinf.bondsectors[j])
        end
        I = sectortype(TensorKit.space(spaces[1]))
        @test all(j -> Hinf.start[j] != Hinf.finish[j], 1:L)
        @test all(j -> Hinf.bondsectors[j][Hinf.start[j]] == unit(I), 1:L)
        @test all(j -> Hinf.bondsectors[j][Hinf.finish[j]] == unit(I), 1:L)
        Ts = mpo_tensors(Hinf)
        @test length(Ts) == L
        for j in 1:L
            @test space(Ts[j], 2) == TensorKit.space(spaces[j])
            @test space(Ts[j], 1) == space(Ts[mod1(j - 1, L)], 4)'
        end
    end
end

@testset "bond dimension does not depend on the unit-cell size" begin
    # a larger cell must reproduce the same MPO site-for-site (translation covariance)
    S = spinvector(BSU2)
    d1 = mpo(opsum(InfiniteChain([BSU2]), dot(S[1], S[2])))
    for L in 2:4
        gen = sum([dot(S[i], S[i + 1]) for i in 1:L])
        dL = mpo(opsum(InfiniteChain(fill(BSU2, L)), gen))
        @test length(dL) == L
        @test all(j -> dL.bondsectors[j] == d1.bondsectors[1], 1:L)
    end

    # and for a model whose range exceeds the cell
    j1j2 = dot(S[1], S[2]) + 0.5 * dot(S[1], S[3])
    a = mpo(opsum(InfiniteChain([BSU2]), j1j2))
    b = mpo(
        opsum(
            InfiniteChain([BSU2, BSU2]),
            dot(S[1], S[2]) + dot(S[2], S[3]) + 0.5 * (dot(S[1], S[3]) + dot(S[2], S[4]))
        )
    )
    @test densedim(a.bondsectors[1]) == 8
    @test all(j -> b.bondsectors[j] == a.bondsectors[1], 1:2)
end

@testset "reference bond dimensions" begin
    S = spinvector(BSU2)
    # textbook infinite Heisenberg: identity, one spin-1 channel in flight, identity
    H = mpo(opsum(InfiniteChain([BSU2]), dot(S[1], S[2])))
    @test H.bondsectors == [[SU2Irrep(0), SU2Irrep(1), SU2Irrep(0)]]
    @test (H.start[1], H.finish[1]) == (1, 3)

    # one extra spin-1 channel per extra neighbour
    for k in 2:4
        gen = sum([dot(S[1], S[1 + r]) for r in 1:k])
        @test densedim(mpo(opsum(InfiniteChain([BSU2]), gen)).bondsectors[1]) == 3k + 2
    end
end

@testset "the answer does not depend on the window size" begin
    for (name, H, spaces) in reference_models()
        L = length(spaces)
        gen = unitcell_terms(opsum(InfiniteChain(spaces), H))
        base = OpSum._infinite_window(gen)
        for nc in (12, 19, 26)
            alt = OpSum._infinite_window(gen; ncells = nc)
            @test alt.bondsectors == base.bondsectors
            @test (alt.start, alt.finish) == (base.start, base.finish)
            @test all(j -> OpSum._entriesequal(alt.Ws[j], base.Ws[j]), 1:L)
        end
    end
end

@testset "input validation" begin
    S = spinvector(BSU2)
    lat = InfiniteChain([BSU2])

    # a translation class written twice (seen across the whole set, so checked when the MPO is formed)
    @test_throws ArgumentError mpo(opsum(lat, dot(S[1], S[2]) + dot(S[2], S[3])))
    # ... but the same pair of bonds is fine on a two-site cell
    @test mpo(
        opsum(InfiniteChain([BSU2, BSU2]), dot(S[1], S[2]) + dot(S[2], S[3]))
    ) isa SymbolicMPO

    # per-term rules are checked on insertion; `Σ_n c·𝟙` does not converge
    @test_throws ArgumentError opsum(lat, dot(S[1], S[2]) + scalarop(1.0, BSU2)[1])
    @test_throws ArgumentError opsum(lat, couple(S[1], S[2]; to = SU2Irrep(1)))
    @test_throws ArgumentError mpo(opsum(lat, dot(S[1], S[2])), IndependentSVD())

    @test_throws ArgumentError InfiniteChain(typeof(BSU2)[])
end

@testset "translate / unitcell_terms / maxspan" begin
    S = spinvector(BSU2)
    H = dot(S[3], S[5])
    @test only(translate(H, 4)).sites == [7, 9]
    @test translate(H, 0) === H
    @test maxspan(H) == 2
    @test termspan(only(H)) == 2
    @test maxspan(S[1]) == 0

    # anchoring shifts the leftmost site into 1:L, by a multiple of L only
    @test only(unitcell_terms(H, 1)).sites == [1, 3]
    @test only(unitcell_terms(translate(H, 1), 2)).sites == [2, 4]
    @test only(unitcell_terms(translate(H, -6), 3)).sites == [3, 5]

    # window_terms keeps exactly the translates that fit
    chain = InfiniteChain([BSU2])
    gen = unitcell_terms(opsum(chain, dot(S[1], S[2])))
    for nc in 1:5
        w = window_terms(gen, nc)
        @test length(w) == max(0, nc - 1)
        @test all(t -> issubset(t.sites, 1:nc), w)
    end
end

@testset "tiled MPO reproduces the enclosed operator densely" begin
    # contract the tiled tensors between the identity-channel boundary vectors against `instantiate`
    S = spinvector(BSU2)
    A, B = element(BTR, 2), element(BTR, 3)
    cases = [
        ("SU2 Heisenberg L=1", dot(S[1], S[2]), [BSU2], 4),
        ("trivial 2-site + on-site L=1", couple(A[1], B[2]) + 0.7 * B[1], [BTR], 4),
        ("SU2 Heisenberg L=2", dot(S[1], S[2]) + dot(S[2], S[3]), [BSU2, BSU2], 2),
    ]
    for (name, H, spaces, ncells) in cases
        L = length(spaces)
        Hinf = mpo(opsum(InfiniteChain(spaces), H))
        N = ncells * L
        sites = [spaces[mod1(j, L)] for j in 1:N]

        Ts = mpo_tensors(Hinf)
        tiled = [Ts[mod1(j, L)] for j in 1:N]
        O = OpSum.contract_open(tiled, Hinf.bondsectors[L], Hinf.start[L], Hinf.finish[L])
        oracle = instantiate(opsum(sites, mpo_terms_window(Hinf, ncells)))
        @test O ≈ oracle || (println("  $name mismatch"); false)
    end
end
