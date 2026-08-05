using Test
using TensorKit
using LinearAlgebra: dot
using OpSum
using OpSum: irrep_mpo, irrep_mpo_tensors, mpo_terms_window, window_terms, unitcell_terms,
    translate, maxspan, termspan, InfiniteMPO, spin, matrixunit, couple, scalarop, instantiate
using OpSum.IrrepTensorOperators: IrrepOperator

# Reference models
# ================
# Each is `(name, generating TermSum, unit-cell spaces)`. `H` is a *generating set*: the operator is
# `Σ_n translate(H, n·L)`, so a nearest-neighbour chain on a two-site cell needs both bonds written
# out, and on a one-site cell exactly one.

const VSU2 = SU2Space(1 // 2 => 1)
const VU1 = U1Space(-1 // 2 => 1, 1 // 2 => 1)
const VTR = ComplexSpace(2)

function su2_models()
    S = spin(VSU2)
    return [
        ("SU2 Heisenberg L=1", dot(S[1], S[2]), [VSU2]),
        ("SU2 Heisenberg L=2", dot(S[1], S[2]) + dot(S[2], S[3]), [VSU2, VSU2]),
        ("SU2 Heisenberg L=3", dot(S[1], S[2]) + dot(S[2], S[3]) + dot(S[3], S[4]), [VSU2, VSU2, VSU2]),
        ("SU2 dimerised L=2", dot(S[1], S[2]) + 0.4 * dot(S[2], S[3]), [VSU2, VSU2]),
        ("SU2 J1-J2 L=1", dot(S[1], S[2]) + 0.5 * dot(S[1], S[3]), [VSU2]),
        (
            "SU2 J1-J2 L=2", dot(S[1], S[2]) + dot(S[2], S[3]) +
                0.5 * (dot(S[1], S[3]) + dot(S[2], S[4])), [VSU2, VSU2],
        ),
        ("SU2 J1-J2-J3 L=1", dot(S[1], S[2]) + 0.5 * dot(S[1], S[3]) + 0.25 * dot(S[1], S[4]), [VSU2]),
        # K = 3: the caterpillar inner line is not fixed by (charges, total)
        ("SU2 3-site chain L=1", couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]), [VSU2]),
    ]
end

function u1_models()
    Sp = matrixunit(VU1, U1Irrep(1 // 2), U1Irrep(-1 // 2))
    Sm = matrixunit(VU1, U1Irrep(-1 // 2), U1Irrep(1 // 2))
    Sz = 0.5 * (
        matrixunit(VU1, U1Irrep(1 // 2), U1Irrep(1 // 2)) -
            matrixunit(VU1, U1Irrep(-1 // 2), U1Irrep(-1 // 2))
    )
    xxz(i, j; Δ = 1.0) =
        0.5 * couple(Sp[i], Sm[j]) + 0.5 * couple(Sm[i], Sp[j]) + Δ * couple(Sz[i], Sz[j])
    return [
        ("U1 XXZ L=1", xxz(1, 2; Δ = 0.7), [VU1]),
        # a charge-0 on-site letter against a two-site tail: the §2.2 pending↔started collision,
        # which on a periodic lattice fires at *every* bond instead of being unreachable
        ("U1 XXZ + hSz L=1", xxz(1, 2) + 0.3 * Sz[1], [VU1]),
        ("U1 XXZ + hSz L=2", xxz(1, 2) + xxz(2, 3) + 0.3 * Sz[1] + 0.3 * Sz[2], [VU1, VU1]),
        ("U1 XXZ + staggered hSz L=2", xxz(1, 2) + xxz(2, 3) + 0.3 * Sz[1] - 0.3 * Sz[2], [VU1, VU1]),
        ("U1 next-nearest hop L=1", couple(Sp[1], Sm[3]) + couple(Sm[1], Sp[3]), [VU1]),
    ]
end

LO(x) = OpSum.SiteOperator(x)

# `ℂ^2` has multiplicity 2 in the trivial sector, so the alphabet is the four matrix units rather than
# `matrixunit` (which needs multiplicity-one sectors)
function trivial_models()
    A, B = LO.(instances(IrrepOperator, VTR)[2:3])
    return [
        ("trivial 2-site L=1", couple(A[1], B[2]), [VTR]),
        # same collision, trivial sector: every bond charge is the unit, so it degenerates to "a
        # pending term's whole content equals a started term's tail"
        ("trivial 2-site + on-site L=1", couple(A[1], B[2]) + 0.7 * B[1], [VTR]),
        ("trivial 3rd-neighbour L=1", couple(A[1], B[2]) + 0.5 * couple(A[1], B[4]), [VTR]),
    ]
end

reference_models() = vcat(su2_models(), u1_models(), trivial_models())

densedim(sec) = sum(dim, sec; init = 0)

# `mpo_terms_window` is a sandwich against `window_terms`, not an equality: a term's reduced
# coefficient can sit up to `R` sites past its own support when its suffix class is shared with a
# longer term (see the `mpo_terms_window` docstring), so paths near the right edge of a finite window
# are missing. Everything produced must be correct, and everything comfortably inside must be present.
function faithful(H, spaces; ncells = 5)
    chain = InfiniteChain(spaces)
    L = length(spaces)
    gen = unitcell_terms(H, L)
    R = maxspan(gen)
    N = ncells * L
    N > R + 1 || error("window too small for the guard")
    # iterating a TermSum canonicalises it, so coincident terms are already summed; `Term`'s `==`
    # and `hash` ignore the coefficient, which is what makes it the dictionary key here
    got = Dict(t => t.coeff for t in mpo_terms_window(irrep_mpo(H, chain), chain, ncells))
    want = Dict(t => t.coeff for t in window_terms(gen, chain, ncells))

    for (t, v) in got                                    # soundness
        haskey(want, t) || return false, "spurious term on sites $(t.sites)"
        want[t] ≈ v || return false, "wrong coefficient on sites $(t.sites)"
    end
    for (t, _) in want                                   # completeness away from the edge
        maximum(t.sites) <= N - R - 1 || continue
        haskey(got, t) || return false, "missing term on sites $(t.sites)"
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
        Hinf = irrep_mpo(H, InfiniteChain(spaces))
        @test length(Hinf) == L
        @test length(Hinf.bondsectors) == L
        for j in 1:L                                     # bond 0 is bond L
            @test size(Hinf.Ws[j], 1) == length(Hinf.bondsectors[mod1(j - 1, L)])
            @test size(Hinf.Ws[j], 2) == length(Hinf.bondsectors[j])
        end
        # both channels exist everywhere, are distinct, and are charge-neutral
        I = sectortype(spaces[1])
        @test all(j -> Hinf.start[j] != Hinf.done[j], 1:L)
        @test all(j -> Hinf.bondsectors[j][Hinf.start[j]] == unit(I), 1:L)
        @test all(j -> Hinf.bondsectors[j][Hinf.done[j]] == unit(I), 1:L)
        # so every bond carries at least the two identity channels
        @test all(j -> length(Hinf.bondsectors[j]) >= 2, 1:L)
    end
end

@testset "bond dimension does not depend on the unit-cell size" begin
    # Writing a translation-invariant model on a larger cell must reproduce the same MPO
    # site-for-site: this is the sharp statement that the compression is translation-covariant, and
    # `L = 1` is where it is hardest because every bond then poses the same problem.
    S = spin(VSU2)
    d1 = irrep_mpo(dot(S[1], S[2]), InfiniteChain([VSU2]))
    for L in 2:4
        gen = sum([dot(S[i], S[i + 1]) for i in 1:L])
        dL = irrep_mpo(gen, InfiniteChain(fill(VSU2, L)))
        @test length(dL) == L
        @test all(j -> dL.bondsectors[j] == d1.bondsectors[1], 1:L)
        @test all(j -> densedim(dL.bondsectors[j]) == 5, 1:L)
    end

    # and the same for a model whose range exceeds the cell
    j1j2 = dot(S[1], S[2]) + 0.5 * dot(S[1], S[3])
    a = irrep_mpo(j1j2, InfiniteChain([VSU2]))
    b = irrep_mpo(
        dot(S[1], S[2]) + dot(S[2], S[3]) + 0.5 * (dot(S[1], S[3]) + dot(S[2], S[4])),
        InfiniteChain([VSU2, VSU2])
    )
    @test densedim(a.bondsectors[1]) == 8
    @test all(j -> b.bondsectors[j] == a.bondsectors[1], 1:2)
end

@testset "reference bond dimensions" begin
    S = spin(VSU2)
    # textbook infinite Heisenberg: identity, one spin-1 channel in flight, identity
    H = irrep_mpo(dot(S[1], S[2]), InfiniteChain([VSU2]))
    @test H.bondsectors == [[SU2Irrep(0), SU2Irrep(1), SU2Irrep(0)]]
    @test (H.start[1], H.done[1]) == (1, 3)
    @test densedim(H.bondsectors[1]) == 5

    # one extra spin-1 channel per extra neighbour
    for (k, D) in ((2, 8), (3, 11), (4, 14))
        gen = sum([dot(S[1], S[1 + r]) for r in 1:k])
        @test densedim(irrep_mpo(gen, InfiniteChain([VSU2])).bondsectors[1]) == 3k + 2
        @test densedim(irrep_mpo(gen, InfiniteChain([VSU2])).bondsectors[1]) == D
    end
end

@testset "the answer does not depend on the window size" begin
    for (name, H, spaces) in reference_models()
        L = length(spaces)
        chain = InfiniteChain(spaces)
        gen = unitcell_terms(H, L)
        base = OpSum._infinite_window(gen, chain)
        for nc in (12, 19, 26)
            alt = OpSum._infinite_window(gen, chain; ncells = nc)
            @test alt.bondsectors == base.bondsectors
            @test (alt.start, alt.done) == (base.start, base.done)
            @test all(j -> OpSum._entriesequal(alt.Ws[j], base.Ws[j]), 1:L)
        end
    end
end

@testset "tensor assembly wraps around" begin
    for (name, H, spaces) in reference_models()
        L = length(spaces)
        lat = InfiniteChain(spaces)
        Hinf = irrep_mpo(H, lat)
        Ts = irrep_mpo_tensors(Hinf, lat)
        @test length(Ts) == L
        # the tensors tile: site 1's left virtual space is site L's right virtual space
        @test space(Ts[1], 1) == space(Ts[L], 4)'
        for j in 1:L
            @test space(Ts[j], 2) == spaces[j]
            @test space(Ts[j], 1) == space(Ts[mod1(j - 1, L)], 4)'
        end
    end
end

@testset "input validation" begin
    S = spin(VSU2)
    lat = InfiniteChain([VSU2])

    # a translation class written twice would be counted twice
    @test_throws ArgumentError irrep_mpo(dot(S[1], S[2]) + dot(S[2], S[3]), lat)
    @test_throws ArgumentError irrep_mpo(dot(S[1], S[3]) + dot(S[4], S[6]), lat)
    # ... but the same pair of bonds is fine on a two-site cell
    @test irrep_mpo(dot(S[1], S[2]) + dot(S[2], S[3]), InfiniteChain([VSU2, VSU2])) isa InfiniteMPO

    # `Σ_n c·𝟙` does not converge
    @test_throws ArgumentError irrep_mpo(dot(S[1], S[2]) + scalarop(1.0, VSU2)[1], lat)
    # a charged term has no translation-invariant running bond charge
    @test_throws ArgumentError irrep_mpo(couple(S[1], S[2]; to = SU2Irrep(1)), lat)
    # the SVD backend has no bond basis that closes on itself
    @test_throws ArgumentError irrep_mpo(dot(S[1], S[2]), lat, SVDBondAlgorithm())

    @test_throws ArgumentError InfiniteChain(typeof(VSU2)[])
    # a unit-cell length mismatch between MPO and lattice
    Hinf = irrep_mpo(dot(S[1], S[2]), lat)
    @test_throws ArgumentError irrep_mpo_tensors(Hinf, InfiniteChain([VSU2, VSU2]))
end

@testset "translate / unitcell_terms / maxspan" begin
    S = spin(VSU2)
    H = dot(S[3], S[5])
    @test only(translate(H, 4)).sites == [7, 9]
    @test translate(H, 0) === H
    @test maxspan(H) == 2
    @test termspan(only(H)) == 2
    @test maxspan(S[1]) == 0

    # anchoring shifts the leftmost site into 1:L, by a multiple of L only
    @test only(unitcell_terms(H, 1)).sites == [1, 3]
    @test only(unitcell_terms(H, 2)).sites == [1, 3]
    @test only(unitcell_terms(translate(H, 1), 2)).sites == [2, 4]
    @test only(unitcell_terms(translate(H, -6), 3)).sites == [3, 5]

    # window_terms keeps exactly the translates that fit
    chain = InfiniteChain([VSU2])
    gen = unitcell_terms(dot(S[1], S[2]), 1)
    for nc in 1:5
        w = window_terms(gen, chain, nc)
        @test length(w) == max(0, nc - 1)
        @test all(t -> issubset(t.sites, 1:nc), w)
    end
end

@testset "tiled MPO reproduces the enclosed operator densely" begin
    # The strongest check available: contract the tiled tensors between the two identity-channel
    # boundary vectors and compare against `instantiate` of the terms the MPO claims to generate.
    # `mpo_terms_window` has already been pinned against `window_terms` above, so this closes the
    # loop from term sum to symmetric tensors.
    S = spin(VSU2)
    A, B = LO.(instances(IrrepOperator, VTR)[2:3])
    cases = [
        ("SU2 Heisenberg L=1", dot(S[1], S[2]), [VSU2], 4),
        ("SU2 J1-J2 L=1", dot(S[1], S[2]) + 0.5 * dot(S[1], S[3]), [VSU2], 4),
        ("trivial 2-site + on-site L=1", couple(A[1], B[2]) + 0.7 * B[1], [VTR], 4),
        ("SU2 Heisenberg L=2", dot(S[1], S[2]) + dot(S[2], S[3]), [VSU2, VSU2], 2),
    ]
    for (name, H, spaces, ncells) in cases
        L = length(spaces)
        lat = InfiniteChain(spaces)
        Hinf = irrep_mpo(H, lat)
        N = ncells * L
        sites = [spaces[mod1(j, L)] for j in 1:N]

        Ts = irrep_mpo_tensors(Hinf, lat)
        tiled = [Ts[mod1(j, L)] for j in 1:N]
        O = OpSum.contract_open(tiled, Hinf.bondsectors[L], Hinf.start[L], Hinf.done[L])
        oracle = instantiate(mpo_terms_window(Hinf, lat, ncells))
        @test O ≈ oracle || (println("  $name mismatch"); false)
    end
end
