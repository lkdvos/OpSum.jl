using Test
using TensorKit
using LinearAlgebra: dot
using OpSum
using OpSum: irrep_mpo, irrep_mpo_tensors, mpo_terms, mpo_terms_window, window_terms,
    unitcell_terms, maxspan, InfiniteMPO, spin, matrixunit, couple, scalarop, instantiate,
    expterm, expand_channels, chain_terms, channelspan, MixedSum, ExpSum, _lower_channels,
    _entriesequal, _infinite_window, storedpairs, ispassthrough
using OpSum.IrrepTensorOperators: IrrepOperator

# Exponentially decaying interactions
# ===================================
# `expterm(t; decay = λ)` stretches the gap before `t`'s exit block geometrically, so the model it adds
# is an infinite family of terms — `Σ_{i<j} λ^{j-i-1} A_i … B_j` and its multi-site generalisations. The
# sweep turns each into a single bond index carrying `λ` on its diagonal, which is what makes the cost
# independent of the interaction range.
#
# What the tests below pin, in order of how sharp they are:
#
#  * the *terms* the MPO generates, coefficient for coefficient, against the explicit expansion
#    (`window_terms` / `chain_terms`), and against an explicitly truncated finite-range model;
#  * the assembled symmetric tensors, contracted down densely against `instantiate`;
#  * the emitted diagonal being `λ · pass-through` as a *letter* — a bare scalar would be silently
#    dropped by `irrep_mpo_tensors`;
#  * bond dimensions, which is where the merge behaviour of the design table is visible.

const VSU2 = SU2Space(1 // 2 => 1)
const VU1 = U1Space(-1 // 2 => 1, 1 // 2 => 1)
const VTR = ComplexSpace(2)

LO(x) = OpSum.SiteOperator(x)
densedim(sec) = sum(dim, sec; init = 0)

S = spin(VSU2)
Sp = matrixunit(VU1, U1Irrep(1 // 2), U1Irrep(-1 // 2))
Sm = matrixunit(VU1, U1Irrep(-1 // 2), U1Irrep(1 // 2))
Sz = 0.5 * (
    matrixunit(VU1, U1Irrep(1 // 2), U1Irrep(1 // 2)) -
        matrixunit(VU1, U1Irrep(-1 // 2), U1Irrep(-1 // 2))
)
xxz(i, j; Δ = 1.0) =
    0.5 * couple(Sp[i], Sm[j]) + 0.5 * couple(Sm[i], Sp[j]) + Δ * couple(Sz[i], Sz[j])
A3, B3 = LO.(instances(IrrepOperator, VTR)[2:3])

# Every model here is a *generating set* on a cell of `length(spaces)` sites, as for a plain `TermSum`.
function reference_models()
    return [
        ("pure exp SU2 L=1", MixedSum(expterm(dot(S[1], S[2]); decay = 0.5)), [VSU2]),
        ("exp + NN SU2 L=1", dot(S[1], S[2]) + expterm(dot(S[1], S[2]); decay = 0.5), [VSU2]),
        ("exp beyond 2nd neighbour", MixedSum(expterm(dot(S[1], S[3]); decay = 0.5)), [VSU2]),
        (
            "two decays", expterm(dot(S[1], S[2]); decay = 0.5) +
                expterm(dot(S[1], S[2]); decay = -0.25), [VSU2],
        ),
        (
            "exp L=2", expterm(dot(S[1], S[2]); decay = 0.5) +
                expterm(dot(S[2], S[3]); decay = 0.5), [VSU2, VSU2],
        ),
        (
            "exp all pairs L=2",
            sum([expterm(dot(S[a], S[a + d]); decay = 0.5) for a in 1:2 for d in 1:2]),
            [VSU2, VSU2],
        ),
        ("exp XXZ U1", MixedSum(expterm(xxz(1, 2); decay = 0.5)), [VU1]),
        ("exp XXZ + field U1", expterm(xxz(1, 2); decay = 0.5) + 0.3 * Sz[1], [VU1]),
        # a charge-neutral but non-trivial string on the stretched gap
        (
            "exp hop with Sᶻ string",
            MixedSum(expterm(couple(Sp[1], Sm[2]); decay = 0.5, string = 2 * Sz)), [VU1],
        ),
        ("exp trivial sector", MixedSum(expterm(couple(A3[1], B3[2]); decay = 0.4)), [VTR]),
        # a two-site entry block (the loop starts after the block completes) …
        (
            "multi-site entry block",
            MixedSum(
                expterm(
                    couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]); decay = 0.5, exitsite = 3
                )
            ), [VSU2],
        ),
        # … and a two-site exit block (the loop exits onto an ordinary finite tail)
        (
            "multi-site exit block",
            MixedSum(
                expterm(
                    couple(couple(S[1], S[3]; to = SU2Irrep(1)), S[4]); decay = 0.5, exitsite = 3
                )
            ), [VSU2],
        ),
    ]
end

# The `mpo_terms_window` sandwich of `research/infinite-mpo.md` §6, with `R` now the range of the
# *shortest* translates: everything produced must be a translate at its exact coefficient, and
# everything comfortably inside the window must be produced. A geometric channel puts its coefficient on
# its entry rather than its exit, so it needs no extra margin of its own.
function faithful(H, spaces; ncells = 6)
    chain = InfiniteChain(spaces)
    L = length(spaces)
    gen = unitcell_terms(H, L)
    R = maxspan(gen)
    N = ncells * L
    got = Dict(t => t.coeff for t in mpo_terms_window(irrep_mpo(H, chain), chain, ncells))
    want = Dict(t => t.coeff for t in window_terms(gen, chain, ncells))

    for (t, v) in got
        haskey(want, t) || return false, "spurious term on sites $(t.sites)"
        want[t] ≈ v ||
            return false, "wrong coefficient on sites $(t.sites): $v vs $(want[t])"
    end
    for (t, _) in want
        maximum(t.sites) <= N - R - 1 || continue
        haskey(got, t) || return false, "missing term on sites $(t.sites)"
    end
    return true, ""
end

@testset "infinite MPO round-trips its exponentially decaying terms" begin
    for (name, H, spaces) in reference_models()
        ok, why = faithful(H, spaces)
        @test ok || (println("  $name: $why"); false)
    end
end

@testset "explicit truncation is the oracle" begin
    # The honest comparison: the same interaction written out as a finite-range sum up to `Rmax`. Every
    # term the truncated model has must appear in the geometric one at the same coefficient, while the
    # geometric bond dimension stays `Rmax`-independent and the truncated one grows linearly.
    λ = 0.5
    ncells = 12
    chain = InfiniteChain([VSU2])
    geo = irrep_mpo(expterm(dot(S[1], S[2]); decay = λ), chain)
    @test densedim(geo.bondsectors[1]) == 5
    got = Dict(t => t.coeff for t in mpo_terms_window(geo, chain, ncells))

    for Rmax in (2, 4, 7)
        gen = unitcell_terms(sum([λ^(r - 1) * dot(S[1], S[1 + r]) for r in 1:Rmax]), 1)
        trunc = irrep_mpo(gen, chain)
        @test densedim(trunc.bondsectors[1]) == 3Rmax + 2      # grows with the range …
        @test densedim(geo.bondsectors[1]) == 5                # … and this one does not

        for t in window_terms(gen, chain, ncells)
            maximum(t.sites) <= ncells - Rmax - 1 || continue
            @test haskey(got, t) && got[t] ≈ t.coeff
        end
    end
end

@testset "the diagonal is λ times the pass-through letter" begin
    # The trap from `research/infinite-mpo.md` §7: `irrep_mpo_tensors` skips entries whose letter is
    # `nothing`, so a channel whose diagonal were a bare scalar would assemble into a silently wrong
    # tensor. λ rides as an edge weight, so the entry is a scaled *letter*.
    λ = 0.4
    H = irrep_mpo(expterm(dot(S[1], S[2]); decay = λ), InfiniteChain([VSU2]))
    W = only(H.Ws)
    diag = [
        (Tuple(idx)[1], only(pairs(op))) for (idx, op) in storedpairs(W)
            if Tuple(idx)[1] == Tuple(idx)[2] && Tuple(idx)[1] ∉ (H.start[1], H.done[1])
    ]
    @test length(diag) == 1
    m, (letter, coeff) = only(diag)
    @test letter !== nothing && ispassthrough(letter)
    @test coeff ≈ λ
    # the channel is the spin-1 index, and it both receives the entry letter and emits the exit letter
    @test H.bondsectors[1][m] == SU2Irrep(1)
    @test any(Tuple(idx) == (H.start[1], m) for (idx, _) in storedpairs(W))
    @test any(Tuple(idx) == (m, H.done[1]) for (idx, _) in storedpairs(W))
end

@testset "merging: what costs a channel and what does not" begin
    # The design table of `research/infinite-mpo.md` §7, as bond dimensions in the trivial sector (every
    # letter is charge-neutral there, so each channel index costs exactly 1).
    ops = LO.(instances(IrrepOperator, VTR)[2:4])
    a, b, c = ops
    lat = InfiniteChain([VTR])
    D(H) = densedim(irrep_mpo(H, lat).bondsectors[1])

    one_channel = D(expterm(couple(a[1], b[2]); decay = 0.5))
    @test one_channel == 3                                        # start + channel + done

    # same λ, same exit, different entry: the suffix class is *the same object* — they merge
    @test D(
        expterm(couple(a[1], b[2]); decay = 0.5) + expterm(couple(c[1], b[2]); decay = 0.5)
    ) == one_channel

    # same entry, different exit: two classes, but they share the entry left vertex
    @test D(
        expterm(couple(a[1], b[2]); decay = 0.5) + expterm(couple(a[1], c[2]); decay = 0.5)
    ) == one_channel + 1

    # different λ: linearly independent channels, so each costs its own index — this is also how a sum
    # of exponentials (a power-law fit) behaves
    @test D(
        expterm(couple(a[1], b[2]); decay = 0.5) + expterm(couple(a[1], b[2]); decay = 0.25)
    ) == one_channel + 1
    for k in 1:4
        H = sum([expterm(couple(a[1], b[2]); decay = 1 / (k + 2)) for k in 1:k])
        @test D(H) == 2 + k
    end
end

@testset "reference bond dimensions" begin
    lat = InfiniteChain([VSU2])
    # a geometric channel *replaces* the in-flight spin-1 multiplet of a nearest-neighbour model rather
    # than adding to it, so exponentially decaying Heisenberg is as cheap as the nearest-neighbour one
    @test densedim(irrep_mpo(expterm(dot(S[1], S[2]); decay = 0.5), lat).bondsectors[1]) == 5
    @test densedim(irrep_mpo(dot(S[1], S[2]), lat).bondsectors[1]) == 5
    # adding the bare nearest-neighbour bond back on top does cost one more spin-1 channel
    @test densedim(
        irrep_mpo(dot(S[1], S[2]) + expterm(dot(S[1], S[2]); decay = 0.5), lat).bondsectors[1]
    ) == 8

    # A channel's *period* is part of its declaration, so writing the same physical model on a larger
    # cell is not free: on an `L`-site cell the same all-pairs interaction needs `L²` channels, whose
    # classes collapse to the `L` "distance to the next legal exit" states — one spin-1 channel each.
    # The terms are identical (checked in the round-trip testset above); the bond dimension is not.
    for L in 1:3
        H = sum([expterm(dot(S[a], S[a + d]); decay = 0.5) for a in 1:L for d in 1:L])
        Hinf = irrep_mpo(H, InfiniteChain(fill(VSU2, L)))
        @test length(H.channels) == L^2
        @test all(j -> densedim(Hinf.bondsectors[j]) == 3L + 2, 1:L)
    end
end

@testset "the answer depends on neither the window size nor the term order" begin
    for (name, H, spaces) in reference_models()
        L = length(spaces)
        chain = InfiniteChain(spaces)
        gen = unitcell_terms(H, L)
        base = _infinite_window(gen, chain)
        for nc in (13, 21)
            alt = _infinite_window(gen, chain; ncells = nc)
            @test alt.bondsectors == base.bondsectors
            @test (alt.start, alt.done) == (base.start, base.done)
            @test all(j -> _entriesequal(alt.Ws[j], base.Ws[j]), 1:L)
        end
    end

    # channel ids are assigned in a content-derived order, so a shuffled sum gives an identical MPO
    parts = [
        expterm(dot(S[1], S[2]); decay = 0.5), expterm(dot(S[1], S[3]); decay = -0.25),
        0.3 * dot(S[1], S[2]),
    ]
    chain1 = InfiniteChain([VSU2])
    base = _infinite_window(unitcell_terms(sum(parts), 1), chain1)
    for perm in (reverse(eachindex(parts)), circshift(collect(eachindex(parts)), 1))
        alt = _infinite_window(unitcell_terms(sum(parts[collect(perm)]), 1), chain1)
        @test alt.bondsectors == base.bondsectors
        @test (alt.start, alt.done) == (base.start, base.done)
        @test _entriesequal(only(alt.Ws), only(base.Ws))
    end
end

@testset "the unit cell closes and reports two identity channels" begin
    for (name, H, spaces) in reference_models()
        L = length(spaces)
        Hinf = irrep_mpo(H, InfiniteChain(spaces))
        I = sectortype(spaces[1])
        @test length(Hinf) == L
        for j in 1:L
            @test size(Hinf.Ws[j], 1) == length(Hinf.bondsectors[mod1(j - 1, L)])
            @test size(Hinf.Ws[j], 2) == length(Hinf.bondsectors[j])
        end
        @test all(j -> Hinf.start[j] != Hinf.done[j], 1:L)
        @test all(j -> Hinf.bondsectors[j][Hinf.start[j]] == unit(I), 1:L)
        @test all(j -> Hinf.bondsectors[j][Hinf.done[j]] == unit(I), 1:L)

        lat = InfiniteChain(spaces)
        Ts = irrep_mpo_tensors(Hinf, lat)
        @test length(Ts) == L
        @test space(Ts[1], 1) == space(Ts[L], 4)'
        @test all(j -> space(Ts[j], 2) == spaces[j], 1:L)
    end
end

@testset "tiled MPO reproduces the enclosed operator densely" begin
    # The strongest available check: contract the tiled tensors between the two identity-channel
    # boundary vectors against `instantiate` of the terms the MPO claims to generate. This is what
    # exercises the pass-through-on-a-*charged*-bond coupler that a channel's diagonal needs.
    cases = [
        ("pure exp trivial", MixedSum(expterm(couple(A3[1], B3[2]); decay = 0.4)), [VTR], 4),
        ("exp + NN SU2", dot(S[1], S[2]) + expterm(dot(S[1], S[2]); decay = 0.5), [VSU2], 4),
        (
            "exp with Sᶻ string",
            MixedSum(expterm(couple(Sp[1], Sm[2]); decay = 0.5, string = 2 * Sz)), [VU1], 4,
        ),
        (
            "exp L=2", expterm(dot(S[1], S[2]); decay = 0.5) +
                expterm(dot(S[2], S[3]); decay = 0.5), [VSU2, VSU2], 2,
        ),
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

@testset "finite chains truncate the geometric sum" begin
    # On a finite chain every site is a possible entry (period 1) and the operator represented is the
    # geometric sum restricted to the chain — `chain_terms` spells it out. States that could no longer
    # complete are pruned, which is what keeps the right boundary one-dimensional.
    cases = [
        ("exp SU2", MixedSum(expterm(dot(S[1], S[2]); decay = 0.5)), fill(VSU2, 5)),
        (
            "exp + NN SU2", sum([dot(S[i], S[i + 1]) for i in 1:4]) +
                expterm(dot(S[1], S[2]); decay = 0.5), fill(VSU2, 5),
        ),
        ("exp gap trivial", MixedSum(expterm(couple(A3[1], B3[3]); decay = 0.4)), fill(VTR, 5)),
        ("exp XXZ U1", MixedSum(expterm(xxz(1, 2); decay = 0.5)), fill(VU1, 4)),
    ]
    for (name, H, sites) in cases
        N = length(sites)
        Ws, secs = irrep_mpo(H, sites)
        @test length(secs[N]) == 1                      # vacuum-terminated, as for a finite TermSum
        want = Dict(t => t.coeff for t in opsum(sites, chain_terms(H, N)))
        got = Dict(t => t.coeff for t in mpo_terms(Ws, secs, sites))
        @test length(got) == length(want)
        @test all(
            haskey(want, t) && want[t] ≈ v for (t, v) in got
        ) || (println("  $name term mismatch"); false)
    end

    # and the terms it stands for are exactly the pairs that fit
    ts = opsum(fill(VSU2, 4), chain_terms(MixedSum(expterm(dot(S[1], S[2]); decay = 0.5)), 4))
    @test sort([t.sites for t in ts]) ==
        [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    @test only(t.coeff for t in ts if t.sites == [1, 4]) ≈
        0.25 * only(t.coeff for t in ts if t.sites == [3, 4])
end

@testset "a channel-free MixedSum is the plain term bag" begin
    # the shared sweep must be untouched when there are no channels
    H = dot(S[1], S[2]) + 0.5 * dot(S[1], S[3])
    a = irrep_mpo(H, InfiniteChain([VSU2]))
    b = irrep_mpo(MixedSum(H), InfiniteChain([VSU2]))
    @test a.bondsectors == b.bondsectors
    @test (a.start, a.done) == (b.start, b.done)
    @test _entriesequal(only(a.Ws), only(b.Ws))

    # a plain term bag binds its lattice with `opsum`; a `MixedSum` cannot (a channel is not a term),
    # so its finite entry point takes the sites directly
    Wa, sa = irrep_mpo(opsum(fill(VSU2, 5), H))
    Wb, sb = irrep_mpo(MixedSum(H), fill(VSU2, 5))
    @test sa == sb
    @test all(i -> _entriesequal(Wa[i], Wb[i]), 1:5)
end

@testset "expterm input validation" begin
    lat = InfiniteChain([VSU2])
    @test_throws ArgumentError expterm(dot(S[1], S[2]); decay = 1.0)      # not summable
    @test_throws ArgumentError expterm(dot(S[1], S[2]); decay = 1.5)
    @test_throws ArgumentError expterm(dot(S[1], S[2]); decay = 0.0)
    @test_throws ArgumentError expterm(S[1]; decay = 0.5)                # no exit block
    @test_throws ArgumentError expterm(dot(S[1], S[2]); decay = 0.5, exitsite = 1)
    @test_throws ArgumentError expterm(dot(S[1], S[2]); decay = 0.5, exitsite = 3)
    # a charged string would make the running bond charge drift along the loop
    @test_throws ArgumentError expterm(couple(Sp[1], Sm[2]); decay = 0.5, string = Sp)
    @test_throws ArgumentError expterm(couple(Sp[1], Sm[2]); decay = 0.5, string = 0.0 * Sz)
    # a charged channel has no translation-invariant running bond charge
    @test_throws ArgumentError irrep_mpo(
        expterm(couple(S[1], S[2]; to = SU2Irrep(1)); decay = 0.5), lat
    )
    # one representative per translation class, exactly as for terms
    @test_throws ArgumentError irrep_mpo(
        expterm(dot(S[1], S[2]); decay = 0.5) + expterm(dot(S[2], S[3]); decay = 0.5), lat
    )
    # … but the same two on a two-site cell is a different, legitimate model
    @test irrep_mpo(
        expterm(dot(S[1], S[2]); decay = 0.5) + expterm(dot(S[2], S[3]); decay = 0.5),
        InfiniteChain([VSU2, VSU2])
    ) isa InfiniteMPO
    # the SVD backend has no notion of a bond index with a diagonal
    @test_throws ArgumentError irrep_mpo(
        expterm(dot(S[1], S[2]); decay = 0.5), lat, SVDBondAlgorithm()
    )
    @test_throws ArgumentError irrep_mpo(
        expterm(dot(S[1], S[2]); decay = 0.5), fill(VSU2, 4), SVDBondAlgorithm()
    )
end

@testset "expterm algebra and expansion" begin
    e = expterm(dot(S[1], S[2]); decay = 0.5)
    @test length(e) == 1
    @test only(keys(e.channels)).exitsite == 2
    @test channelspan(only(keys(e.channels))) == 1
    @test maxspan(e) == 1

    # scaling and adding behave like a TermSum's
    @test only(values((2 * e).channels)) ≈ 2 * only(values(e.channels))
    @test isempty((e - e).channels)
    H = dot(S[1], S[2]) + e
    @test H isa MixedSum
    @test length(H.terms.terms) == 1 && length(H.channels) == 1
    @test length((H + e).channels) == 1                      # same descriptor, coefficients add
    @test only(values((H + e).channels.channels)) ≈ 2 * only(values(e.channels))

    # a composite representative distributes into one channel per term, as `couple` does
    @test length(expterm(xxz(1, 2); decay = 0.5)) == length(xxz(1, 2).terms)

    # the expansion: geometric coefficients, and only the translates that fit
    ts = expand_channels(e, 1, 4)
    coeff(sites) = only(t.coeff for t in ts if t.sites == sites)
    @test length(ts) == 6
    @test coeff([1, 3]) ≈ 0.5 * coeff([1, 2])
    @test coeff([1, 4]) ≈ 0.25 * coeff([1, 2])
    @test coeff([2, 3]) ≈ coeff([1, 2])
    # with a period of 2 only the phase-matched pairs survive, and each extra cell costs λ²
    ts2 = expand_channels(e, 2, 6)
    @test sort([t.sites for t in ts2]) == [[1, 2], [1, 4], [1, 6], [3, 4], [3, 6], [5, 6]]
    c2(sites) = only(t.coeff for t in ts2 if t.sites == sites)
    @test c2([1, 4]) ≈ 0.25 * c2([1, 2])

    # the lowered automaton: one cyclic state per period, plus the mandatory waits of a longer gap
    @test length(only(_lower_channels(e, 1)).states) == 1
    @test length(only(_lower_channels(e, 3)).states) == 3
    gap = expterm(dot(S[1], S[4]); decay = 0.5)
    c = only(_lower_channels(gap, 2))
    @test length(c.states) == 3                              # δ = 3 (wait), 2 and 1 (cyclic)
    @test count(st -> st.cyclic, c.states) == 2
    # channels that must merge share their state names, channels with a different λ do not
    same = _lower_channels(
        expterm(couple(A3[1], B3[2]); decay = 0.5) + expterm(couple(A3[2], B3[3]); decay = 0.5), 1
    )
    @test allequal(only(ch.states).name for ch in same)
    diff = _lower_channels(
        expterm(couple(A3[1], B3[2]); decay = 0.5) + expterm(couple(A3[1], B3[2]); decay = 0.25), 1
    )
    @test allunique(only(ch.states).name for ch in diff)
end
