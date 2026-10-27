using Test
using TensorKit
using LinearAlgebra: dot
using OpSum
using OpSum: mpo, mpo_tensors, jordan_mpo_tensors, mpo_terms, mpo_terms_window, window_terms,
    unitcell_terms, maxspan, translate, couple, scalarop, instantiate,
    expterm, expand_channels, chain_terms, channelspan, ExpSum, lower_channels,
    _entriesequal, _infinite_window, storedpairs, ispassthrough
using OpSum: Letter

const VSU2 = SU2Space(1 // 2 => 1)
const VU1 = U1Space(-1 // 2 => 1, 1 // 2 => 1)
const VTR = ComplexSpace(2)

include(joinpath(@__DIR__, "testutils.jl"))   # unitbasis, element, matrixunit, spinvector
const BSU2 = unitbasis(VSU2)
const BU1 = spin_basis(U1Irrep)      # stock basis: Sᶻ is one atomic letter, so `couple(Sz, Sz)` is legal
const BTR = unitbasis(VTR)

LO(x) = OpSum.SiteOperator(x)
densedim(sec) = sum(dim, sec; init = 0)

S = spinvector(BSU2)
Sp = matrixunit(BU1, U1Irrep(1 // 2), U1Irrep(-1 // 2))
Sm = matrixunit(BU1, U1Irrep(-1 // 2), U1Irrep(1 // 2))
Sz = spin_ops(U1Irrep).S_z
xxz(i, j; Δ = 1.0) =
    0.5 * couple(Sp[i], Sm[j]) + 0.5 * couple(Sm[i], Sp[j]) + Δ * couple(Sz[i], Sz[j])
A3, B3 = element(BTR, 2), element(BTR, 3)

# Every model is a generating set on a cell of `length(spaces)` sites, on the infinite chain of that cell.
function reference_models()
    m(spaces, parts...) = opsum(InfiniteChain(spaces), parts...)
    return [
        ("pure exp SU2 L=1", m([BSU2], expterm(dot(S[1], S[2]); decay = 0.5)), [BSU2]),
        (
            "exp + NN SU2 L=1",
            m([BSU2], dot(S[1], S[2]), expterm(dot(S[1], S[2]); decay = 0.5)), [BSU2],
        ),
        ("exp beyond 2nd neighbour", m([BSU2], expterm(dot(S[1], S[3]); decay = 0.5)), [BSU2]),
        (
            "two decays",
            m(
                [BSU2], expterm(dot(S[1], S[2]); decay = 0.5),
                expterm(dot(S[1], S[2]); decay = -0.25)
            ), [BSU2],
        ),
        (
            "exp L=2",
            m(
                [BSU2, BSU2], expterm(dot(S[1], S[2]); decay = 0.5),
                expterm(dot(S[2], S[3]); decay = 0.5)
            ), [BSU2, BSU2],
        ),
        (
            "exp all pairs L=2",
            m([BSU2, BSU2], (expterm(dot(S[a], S[a + d]); decay = 0.5) for a in 1:2 for d in 1:2)),
            [BSU2, BSU2],
        ),
        ("exp XXZ U1", m([BU1], expterm(xxz(1, 2); decay = 0.5)), [BU1]),
        (
            "exp XXZ + field U1",
            m([BU1], expterm(xxz(1, 2); decay = 0.5), 0.3 * Sz[1]), [BU1],
        ),
        # a charge-neutral but non-trivial string on the stretched gap
        (
            "exp hop with Sᶻ string",
            m([BU1], expterm(couple(Sp[1], Sm[2]); decay = 0.5, string = 2 * Sz)), [BU1],
        ),
        ("exp trivial sector", m([BTR], expterm(couple(A3[1], B3[2]); decay = 0.4)), [BTR]),
        # a two-site entry block (the loop starts after the block completes) …
        (
            "multi-site entry block",
            m(
                [BSU2],
                expterm(
                    couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]); decay = 0.5, exitsite = 3
                )
            ), [BSU2],
        ),
        # … and a two-site exit block (the loop exits onto an ordinary finite tail)
        (
            "multi-site exit block",
            m(
                [BSU2],
                expterm(
                    couple(couple(S[1], S[3]; to = SU2Irrep(1)), S[4]); decay = 0.5, exitsite = 3
                )
            ), [BSU2],
        ),
    ]
end

# Everything the window produces must be a translate at its exact coefficient, and everything comfortably
# inside the window must be produced.
function faithful(H, spaces; ncells = 6)
    L = length(spaces)
    gen = unitcell_terms(H)
    R = maxspan(gen)
    N = ncells * L
    got = termdict(mpo_terms_window(mpo(H), ncells))
    want = termdict(window_terms(gen, ncells))

    for (t, v) in got
        haskey(want, t) || return false, "spurious term on sites $(first(t))"
        want[t] ≈ v ||
            return false, "wrong coefficient on sites $(first(t)): $v vs $(want[t])"
    end
    for (t, _) in want
        maximum(first(t)) <= N - R - 1 || continue
        haskey(got, t) || return false, "missing term on sites $(first(t))"
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
    # A finite-range sum up to `Rmax`: its terms appear in the geometric one, whose bond dimension is `Rmax`-independent.
    λ = 0.5
    ncells = 12
    chain = InfiniteChain([BSU2])
    geo = mpo(opsum(chain, expterm(dot(S[1], S[2]); decay = λ)))
    @test densedim(geo.bondsectors[1]) == 5
    got = termdict(mpo_terms_window(geo, ncells))

    for Rmax in (2, 4, 7)
        gen = unitcell_terms(sum([λ^(r - 1) * dot(S[1], S[1 + r]) for r in 1:Rmax]), 1)
        trunc = mpo(opsum(chain, gen))
        @test densedim(trunc.bondsectors[1]) == 3Rmax + 2      # grows with the range …
        @test densedim(geo.bondsectors[1]) == 5                # … and this one does not

        for t in window_terms(gen, chain, ncells)
            maximum(t.sites) <= ncells - Rmax - 1 || continue
            @test haskey(got, OpSum.support(t)) && got[OpSum.support(t)] ≈ t.coeff
        end
    end
end

@testset "the diagonal is λ times the pass-through letter" begin
    # `mpo_tensors` skips letter-less entries, so a bare-scalar diagonal would be silently dropped.
    λ = 0.4
    H = mpo(opsum(InfiniteChain([BSU2]), expterm(dot(S[1], S[2]); decay = λ)))
    W = only(H.Ws)
    diag = [
        (Tuple(idx)[1], only(pairs(op))) for (idx, op) in storedpairs(W)
            if Tuple(idx)[1] == Tuple(idx)[2] && Tuple(idx)[1] ∉ (H.start[1], H.finish[1])
    ]
    @test length(diag) == 1
    m, (letter, coeff) = only(diag)
    @test letter !== nothing && ispassthrough(letter)
    @test coeff ≈ λ
    # the channel is the spin-1 index, and it both receives the entry letter and emits the exit letter
    @test H.bondsectors[1][m] == SU2Irrep(1)
    @test any(Tuple(idx) == (H.start[1], m) for (idx, _) in storedpairs(W))
    @test any(Tuple(idx) == (m, H.finish[1]) for (idx, _) in storedpairs(W))
end

@testset "merging: what costs a channel and what does not" begin
    # Each channel index costs 1 (trivial sector).
    ops = [element(BTR, n) for n in 2:4]
    a, b, c = ops
    lat = InfiniteChain([BTR])
    D(H) = densedim(mpo(opsum(lat, H)).bondsectors[1])

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
    lat = InfiniteChain([BSU2])
    # a geometric channel *replaces* the in-flight spin-1 multiplet of a nearest-neighbour model rather
    # than adding to it, so exponentially decaying Heisenberg is as cheap as the nearest-neighbour one
    @test densedim(mpo(opsum(lat, dot(S[1], S[2]))).bondsectors[1]) == 5      # = the exponential one (see above)
    # adding the bare nearest-neighbour bond back on top does cost one more spin-1 channel
    @test densedim(
        mpo(opsum(lat, dot(S[1], S[2]), expterm(dot(S[1], S[2]); decay = 0.5))).bondsectors[1]
    ) == 8

    # A channel's period is part of its declaration: on an `L`-site cell the all-pairs interaction needs
    # `L²` channels, collapsing to `L` spin-1 channels.
    for L in 1:3
        H = sum([expterm(dot(S[a], S[a + d]); decay = 0.5) for a in 1:L for d in 1:L])
        Hinf = mpo(opsum(InfiniteChain(fill(BSU2, L)), H))
        @test length(H.channels) == L^2
        @test all(j -> densedim(Hinf.bondsectors[j]) == 3L + 2, 1:L)
    end
end

@testset "the answer depends on neither the window size nor the term order" begin
    for (name, H, spaces) in reference_models()
        L = length(spaces)
        gen = unitcell_terms(H)
        base = _infinite_window(gen)
        for nc in (13, 21)
            alt = _infinite_window(gen; ncells = nc)
            @test alt.bondsectors == base.bondsectors
            @test (alt.start, alt.finish) == (base.start, base.finish)
            @test all(j -> _entriesequal(alt.Ws[j], base.Ws[j]), 1:L)
        end
    end

    # channel ids are assigned in a content-derived order, so a shuffled sum gives an identical MPO
    parts = [
        expterm(dot(S[1], S[2]); decay = 0.5), expterm(dot(S[1], S[3]); decay = -0.25),
        0.3 * dot(S[1], S[2]),
    ]
    chain1 = InfiniteChain([BSU2])
    base = _infinite_window(unitcell_terms(opsum(chain1, parts...)))
    for perm in (reverse(eachindex(parts)), circshift(collect(eachindex(parts)), 1))
        alt = _infinite_window(unitcell_terms(opsum(chain1, parts[collect(perm)]...)))
        @test alt.bondsectors == base.bondsectors
        @test (alt.start, alt.finish) == (base.start, base.finish)
        @test _entriesequal(only(alt.Ws), only(base.Ws))
    end
end

@testset "ExpSum: exact equality, hash and content order" begin
    e1 = expterm(dot(S[1], S[2]); decay = 0.5)
    e2 = expterm(dot(S[1], S[3]); decay = -0.25)
    @test e1 + e2 == e2 + e1 && hash(e1 + e2) == hash(e2 + e1)
    @test e1 + e2 != e1 && e1 != 2 * e1 && e1 == copy(e1)
    @test e1 - e1 == ExpSum{SU2Irrep}() && hash(e1 - e1) == hash(ExpSum{SU2Irrep}())
    @test e1 + e2 ≈ e2 + e1 && !(e1 ≈ 2 * e1) && !(e1 ≈ e2)

    # the key's content order is total and independent of the writing order
    k1, k2 = only(keys(e1.channels)), only(keys(e2.channels))
    @test isless(k1, k2) != isless(k2, k1)
    @test sort([k2, k1]) == sort([k1, k2])
    @test k1 == translate(k1, 0) && translate(k1, 2) != k1
    # the string is part of the key; the default string is the identity
    p = expterm(couple(Sp[1], Sm[2]); decay = 0.5)
    @test p == expterm(couple(Sp[1], Sm[2]); decay = 0.5, string = scalarop(1, U1Irrep))
    q = expterm(couple(Sp[1], Sm[2]); decay = 0.5, string = 2 * Sz)
    @test p != q && hash(p) != hash(q)
    @test sort([only(keys(q.channels)), only(keys(p.channels))]) == sort([only(keys(p.channels)), only(keys(q.channels))])
    # a cancelling coefficient drops the channel instead of leaving a zero
    @test isempty(OpSum.addchannel!(copy(e1), k1, -only(values(e1.channels))))
end

@testset "the unit cell closes and reports two identity channels" begin
    for (name, H, spaces) in reference_models()
        L = length(spaces)
        Hinf = mpo(H)
        I = sectortype(TensorKit.space(spaces[1]))
        @test length(Hinf) == L
        for j in 1:L
            @test size(Hinf.Ws[j], 1) == length(Hinf.bondsectors[mod1(j - 1, L)])
            @test size(Hinf.Ws[j], 2) == length(Hinf.bondsectors[j])
        end
        @test all(j -> Hinf.start[j] != Hinf.finish[j], 1:L)
        @test all(j -> Hinf.bondsectors[j][Hinf.start[j]] == unit(I), 1:L)
        @test all(j -> Hinf.bondsectors[j][Hinf.finish[j]] == unit(I), 1:L)

        Ts = mpo_tensors(Hinf)
        @test length(Ts) == L
        @test space(Ts[1], 1) == space(Ts[L], 4)'
        @test all(j -> space(Ts[j], 2) == TensorKit.space(spaces[j]), 1:L)
    end
end

@testset "tiled MPO reproduces the enclosed operator densely" begin
    # Contract the tiled tensors between the identity-channel boundary vectors against `instantiate`.
    cases = [
        ("pure exp trivial", [expterm(couple(A3[1], B3[2]); decay = 0.4)], [BTR], 4),
        (
            "exp with Sᶻ string",
            [expterm(couple(Sp[1], Sm[2]); decay = 0.5, string = 2 * Sz)], [BU1], 4,
        ),
        (
            "exp L=2",
            [expterm(dot(S[1], S[2]); decay = 0.5), expterm(dot(S[2], S[3]); decay = 0.5)],
            [BSU2, BSU2], 2,
        ),
    ]
    for (name, H, spaces, ncells) in cases
        L = length(spaces)
        Hinf = mpo(opsum(InfiniteChain(spaces), H...))
        N = ncells * L
        sites = [spaces[mod1(j, L)] for j in 1:N]
        Ts = mpo_tensors(Hinf)
        tiled = [Ts[mod1(j, L)] for j in 1:N]
        O = OpSum.contract_open(tiled, Hinf.bondsectors[L], Hinf.start[L], Hinf.finish[L])
        oracle = instantiate(opsum(sites, mpo_terms_window(Hinf, ncells)))
        @test O ≈ oracle || (println("  $name mismatch"); false)
    end
end

@testset "finite chains truncate the geometric sum" begin
    cases = [
        (
            "exp + NN SU2",
            [(dot(S[i], S[i + 1]) for i in 1:4), expterm(dot(S[1], S[2]); decay = 0.5)],
            fill(BSU2, 5),
        ),
        ("exp XXZ U1", [expterm(xxz(1, 2); decay = 0.5)], fill(BU1, 4)),
    ]
    for (name, parts, sites) in cases
        N = length(sites)
        H = opsum(FiniteChain(sites), parts...)
        W = mpo(H)
        Ws, secs = W.Ws, W.bondsectors
        @test length(secs[N]) == 1                      # vacuum-terminated, as on a FiniteChain
        @test length(W.start) == N && length(W.finish) == N
        @test length(jordan_mpo_tensors(W)) == N
        want = termdict(chain_terms(H))
        got = termdict(mpo_terms(W))
        @test length(got) == length(want)
        @test all(
            haskey(want, t) && want[t] ≈ v for (t, v) in got
        ) || (println("  $name term mismatch"); false)
        @test islossless(H)
    end

    # and the terms it stands for are exactly the pairs that fit
    ts = chain_terms(opsum(FiniteChain(BSU2, 4), expterm(dot(S[1], S[2]); decay = 0.5)))
    @test sort([t.sites for t in ts]) ==
        [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    @test only(t.coeff for t in ts if t.sites == [1, 4]) ≈
        0.25 * only(t.coeff for t in ts if t.sites == [3, 4])
end

@testset "a channel-free OperatorSum is the plain term bag" begin
    # the shared sweep must be untouched when there are no channels
    H = dot(S[1], S[2]) + 0.5 * dot(S[1], S[3])
    chain = InfiniteChain([BSU2])
    a = mpo(opsum(chain, H))
    b = _infinite_window(unitcell_terms(opsum(chain, H, ExpSum{SU2Irrep}())))
    @test a.bondsectors == b.bondsectors
    @test (a.start, a.finish) == (b.start, b.finish)
    @test _entriesequal(only(a.Ws), only(b.Ws))
end

@testset "expterm input validation" begin
    lat = InfiniteChain([BSU2])
    @test_throws ArgumentError expterm(dot(S[1], S[2]); decay = 1.0)      # not summable
    @test_throws ArgumentError expterm(dot(S[1], S[2]); decay = 0.0)
    @test_throws ArgumentError expterm(S[1]; decay = 0.5)                # no exit block
    @test_throws ArgumentError expterm(dot(S[1], S[2]); decay = 0.5, exitsite = 1)
    @test_throws ArgumentError expterm(dot(S[1], S[2]); decay = 0.5, exitsite = 3)
    # a charged string would make the running bond charge drift along the loop
    @test_throws ArgumentError expterm(couple(Sp[1], Sm[2]); decay = 0.5, string = Sp)
    @test_throws ArgumentError expterm(couple(Sp[1], Sm[2]); decay = 0.5, string = 0.0 * Sz)
    # a charged channel has no translation-invariant running bond charge; refused as it enters
    @test_throws ArgumentError opsum(
        lat, expterm(couple(S[1], S[2]; to = SU2Irrep(1)); decay = 0.5)
    )
    # one representative per translation class, exactly as for terms
    @test_throws ArgumentError mpo(
        opsum(lat, expterm(dot(S[1], S[2]); decay = 0.5), expterm(dot(S[2], S[3]); decay = 0.5))
    )
    # … but the same two on a two-site cell is a different, legitimate model
    @test mpo(
        opsum(
            InfiniteChain([BSU2, BSU2]),
            expterm(dot(S[1], S[2]); decay = 0.5), expterm(dot(S[2], S[3]); decay = 0.5)
        )
    ) isa SymbolicMPO
    # the SVD backend has no notion of a bond index with a diagonal
    @test_throws ArgumentError mpo(
        opsum(lat, expterm(dot(S[1], S[2]); decay = 0.5)), IndependentSVD()
    )
    @test_throws ArgumentError mpo(
        opsum(FiniteChain(BSU2, 4), expterm(dot(S[1], S[2]); decay = 0.5)), IndependentSVD()
    )
end

@testset "expterm algebra and expansion" begin
    e = expterm(dot(S[1], S[2]); decay = 0.5)
    @test length(e) == 1
    @test only(keys(e.channels)).exitsite == 2
    @test channelspan(only(keys(e.channels))) == 1

    # scaling and adding behave like a term bag's
    @test only(values((2 * e).channels)) ≈ 2 * only(values(e.channels))
    @test isempty((e - e).channels)
    H = opsum(FiniteChain(BSU2, 3), dot(S[1], S[2]), e)
    @test H isa OperatorSum
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
    @test length(only(lower_channels(e, 1)).states) == 1
    @test length(only(lower_channels(e, 3)).states) == 3
    gap = expterm(dot(S[1], S[4]); decay = 0.5)
    c = only(lower_channels(gap, 2))
    @test length(c.states) == 3                              # δ = 3 (wait), 2 and 1 (cyclic)
    @test count(st -> st.cyclic, c.states) == 2
    # channels that must merge share their state names, channels with a different λ do not
    same = lower_channels(
        expterm(couple(A3[1], B3[2]); decay = 0.5) + expterm(couple(A3[2], B3[3]); decay = 0.5), 1
    )
    @test allequal(only(ch.states).name for ch in same)
    diff = lower_channels(
        expterm(couple(A3[1], B3[2]); decay = 0.5) + expterm(couple(A3[1], B3[2]); decay = 0.25), 1
    )
    @test allunique(only(ch.states).name for ch in diff)
end
