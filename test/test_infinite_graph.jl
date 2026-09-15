using Test
using TensorKit
using LinearAlgebra: dot
using OpSum
using OpSum: spin, couple, matrixunit, translate, unitcell_terms, window_terms, maxspan,
    ITOTermTable, ITOGraph, _irrep_sweep, _rel_suffix_ids, _suffix_ids, _rdesc,
    _fixedpoint_cell, _entriesequal, _cellsequal, _identity_channels, _infinite_window,
    _at_site!, arity, nterms

# The two halves of the canonicalisation that makes the sweep translation-covariant:
#
#   * `_rel_suffix_ids` / `_rdesc` — a suffix class is named by *shape plus distance from the bond*
#     rather than by absolute position, so a class and its translate one unit cell later are the same
#     object;
#   * the canonical ordering itself — right vertices renumbered by that name, adjacency lists sorted,
#     and the assembled bond ordered by `(link, key)` then by class name.
#
# Neither changes any bond dimension: what they remove is the *history dependence* of which minimum
# vertex cover Hopcroft–Karp happens to find. On a finite chain that is invisible (all minimum covers
# are equally good); on a periodic lattice it decides whether the unit cell closes at all.

const VSU2 = SU2Space(1 // 2 => 1)
const VU1 = U1Space(-1 // 2 => 1, 1 // 2 => 1)

@testset "relative suffix ids are translation invariant" begin
    S = spin(VSU2)
    # the same two-site term at four different positions
    H = sum([dot(S[i], S[i + 2]) for i in 1:4])
    tt = ITOTermTable(opsum(fill(VSU2, 8), H))
    rel = _rel_suffix_ids(tt)
    abs_ = _suffix_ids(tt)
    K, M = arity(tt), nterms(tt)
    @test size(rel) == (K + 1, M)

    # every term is a translate of every other, so all shape ids agree column by column …
    for j in 1:K
        @test allequal(rel[j, t] for t in 1:M)
    end
    # … whereas the position-keyed ids are all distinct (which is exactly why a second naming was
    # needed: `_suffix_ids` can never see two bonds one cell apart as posing the same problem)
    @test allunique(abs_[1, t] for t in 1:M)
    # exhausted stays 0 in both
    @test all(iszero, rel[K + 1, :])
    @test all(iszero, abs_[K + 1, :])
end

@testset "live class names repeat with the unit cell" begin
    # The runtime half of the same statement: the *set of names* of the live suffix classes at bond `i`
    # equals the one at bond `i + L` once the sweep is in the bulk. Recorded straight after each site
    # step (which leaves the cursors at site `i`, plus the classes injected for `i + 1` — themselves
    # periodic, so the comparison still holds).
    S = spin(VSU2)
    for (L, H, spaces) in [
            (1, dot(S[1], S[2]) + 0.5 * dot(S[1], S[3]), [VSU2]),
            (2, dot(S[1], S[2]) + dot(S[2], S[3]), [VSU2, VSU2]),
        ]
        gen = unitcell_terms(H, L)
        ncells = 12
        N = ncells * L
        tt = ITOTermTable(window_terms(gen, InfiniteChain(spaces), ncells))
        g = ITOGraph(tt, N)
        names = map(1:N) do i
            _at_site!(g, i)
            return sort([_rdesc(g, r, i) for r in eachindex(g.rrepr)])
        end
        guard = 4L + maxspan(gen)
        @test all(i -> names[i] == names[i + L], (guard + 1):(N - guard))
        # and the live set stays bounded independently of the window length
        @test allequal(length.(names[(guard + 1):(N - guard)]))
    end
end

@testset "bulk bonds of an unrolled sweep are literally equal" begin
    # The property `_fixedpoint_cell` rests on, and the one that fails without canonicalisation: bulk
    # site tensors are equal entry for entry, not merely of equal size.
    S = spin(VSU2)
    for (name, H, L, spaces) in [
            ("Heisenberg L=1", dot(S[1], S[2]), 1, [VSU2]),
            ("J1-J2 L=1", dot(S[1], S[2]) + 0.5 * dot(S[1], S[3]), 1, [VSU2]),
            ("Heisenberg L=2", dot(S[1], S[2]) + dot(S[2], S[3]), 2, [VSU2, VSU2]),
        ]
        gen = unitcell_terms(H, L)
        R = maxspan(gen)
        ncells = 14
        N = ncells * L
        tt = ITOTermTable(window_terms(gen, InfiniteChain(spaces), ncells))
        Ws, secs = _irrep_sweep(tt, N, VertexCover())

        c = _fixedpoint_cell(Ws, secs, L, R)
        @test 0 < c < ncells - 1
        # every cell from `c` up to the right-boundary region repeats
        for cc in c:(ncells - 4)
            @test _cellsequal(Ws, L, cc)
            @test secs[(cc - 1) * L] == secs[cc * L]
        end
    end
end

@testset "the sweep does not depend on the order terms were written in" begin
    # The sharp statement of what canonicalisation buys. Building the same Hamiltonian from a shuffled
    # term list gives a different append order, and `canonicalize!` sorts the terms but the sweep still
    # derives its right-vertex ids and adjacency order from where each class was first encountered.
    # Before canonicalisation that could change which minimum cover was found; now the reduced MPO is
    # identical entry for entry.
    S = spin(VSU2)
    Sp = matrixunit(VU1, U1Irrep(1 // 2), U1Irrep(-1 // 2))
    Sm = matrixunit(VU1, U1Irrep(-1 // 2), U1Irrep(1 // 2))
    Sz = 0.5 * (
        matrixunit(VU1, U1Irrep(1 // 2), U1Irrep(1 // 2)) -
            matrixunit(VU1, U1Irrep(-1 // 2), U1Irrep(-1 // 2))
    )
    cases = [
        (
            "SU2 J1-J2", 1, VSU2,
            [dot(S[1], S[2]), 0.5 * dot(S[1], S[3]), 0.25 * dot(S[1], S[4])],
        ),
        (
            "U1 XXZ + field", 1, VU1,
            [
                0.5 * couple(Sp[1], Sm[2]), 0.5 * couple(Sm[1], Sp[2]),
                couple(Sz[1], Sz[2]), 0.3 * Sz[1],
            ],
        ),
        (
            "SU2 dimerised", 2, VSU2,
            [dot(S[1], S[2]), 0.4 * dot(S[2], S[3]), 0.2 * dot(S[1], S[3])],
        ),
    ]
    for (name, L, V, parts) in cases
        chain = InfiniteChain(fill(V, L))
        base = _infinite_window(unitcell_terms(sum(parts), L), chain)
        for perm in (reverse(eachindex(parts)), circshift(collect(eachindex(parts)), 1))
            alt = _infinite_window(unitcell_terms(sum(parts[collect(perm)]), L), chain)
            @test alt.bondsectors == base.bondsectors
            @test (alt.start, alt.done) == (base.start, base.done)
            @test all(j -> _entriesequal(alt.Ws[j], base.Ws[j]), 1:L)
        end
    end
end

@testset "identity-channel detection" begin
    S = spin(VSU2)
    H = _infinite_window(unitcell_terms(dot(S[1], S[2]), 1), InfiniteChain([VSU2]))
    @test _identity_channels(H.Ws, H.bondsectors) == (H.start, H.done)

    # the start channel's column and the done channel's row each hold exactly one stored entry
    W = H.Ws[1]
    s, d = H.start[1], H.done[1]
    @test count(idx -> Tuple(idx)[2] == s, first.(collect(OpSum.storedpairs(W)))) == 1
    @test count(idx -> Tuple(idx)[1] == d, first.(collect(OpSum.storedpairs(W)))) == 1

    # a cell whose only entry is a charged letter has no closed pass-through backbone at all
    LOp = OpSum.SiteOperator{SU2Irrep}
    d = OpSum.Dictionary{CartesianIndex{2}, LOp}()
    OpSum.increaseindex!(
        d, CartesianIndex(1, 1), OpSum.SiteOperator(OpSum.IrrepOperator(SU2Irrep(1), 1))
    )
    @test_throws ArgumentError _identity_channels(
        [OpSum.sparse_from_dict(d, (1, 1))], [[SU2Irrep(1)]]
    )
end

@testset "canonicalisation leaves the finite path's bond dimensions alone" begin
    # A guard on the shared code: the finite sweep is exercised in full by `test_irrep_graph.jl`, but
    # pin the textbook numbers here too so a regression in `_canonicalise_bond!` is attributed here.
    S = spin(VSU2)
    for N in (4, 6, 8)
        H = sum([dot(S[i], S[i + 1]) for i in 1:(N - 1)])
        _, secs = OpSum.irrep_mpo(opsum(fill(VSU2, N), H))
        dense = [sum(dim, sec) for sec in secs]
        @test dense[1] == 4                       # spin-1 channel + identity
        @test all(==(5), dense[2:(N - 2)])        # bulk
        @test dense[N] == 1                       # closed
    end
end
