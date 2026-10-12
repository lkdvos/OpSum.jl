using Test
using TensorKit
using LinearAlgebra: dot
using OpSum
using OpSum: couple, translate, unitcell_terms, window_terms, maxspan,
    ITOTermTable, ITOGraph, _irrep_sweep, _rel_suffix_ids, _suffix_ids, _rdesc,
    _fixedpoint_cell, _entriesequal, _cellsequal, _identity_channels, _infinite_window,
    _at_site!, arity, nterms

# The canonicalisation that makes the sweep translation-covariant: suffix classes are named by shape plus
# distance from the bond (`_rel_suffix_ids`/`_rdesc`), and right vertices, adjacency lists and bonds are
# ordered by that name. No bond dimension changes; it removes the history dependence of which minimum
# cover is found, which on a periodic lattice decides whether the unit cell closes.

const VSU2 = SU2Space(1 // 2 => 1)
const VU1 = U1Space(-1 // 2 => 1, 1 // 2 => 1)

include(joinpath(@__DIR__, "testutils.jl"))   # unitbasis, matrixunit, spinvector
const BSU2 = unitbasis(VSU2)
const BU1 = spin_basis(U1Irrep)   # stock basis: Sᶻ is one atomic letter

@testset "relative suffix ids are translation invariant" begin
    S = spinvector(BSU2)
    H = sum([dot(S[i], S[i + 2]) for i in 1:4])
    tt = ITOTermTable(opsum(H), 8)
    rel = _rel_suffix_ids(tt)
    abs_ = _suffix_ids(tt)
    K, M = arity(tt), nterms(tt)
    @test size(rel) == (K + 1, M)

    # all terms are translates, so shape ids agree column by column …
    for j in 1:K
        @test allequal(rel[j, t] for t in 1:M)
    end
    # … whereas the position-keyed ids are all distinct
    @test allunique(abs_[1, t] for t in 1:M)
    @test all(iszero, rel[K + 1, :])
    @test all(iszero, abs_[K + 1, :])
end

@testset "live class names repeat with the unit cell" begin
    # the set of live suffix-class names at bond `i` equals the one at `i + L` in the bulk
    S = spinvector(BSU2)
    for (L, H, spaces) in [
            (1, dot(S[1], S[2]) + 0.5 * dot(S[1], S[3]), [BSU2]),
            (2, dot(S[1], S[2]) + dot(S[2], S[3]), [BSU2, BSU2]),
        ]
        gen = unitcell_terms(H, L)
        ncells = 12
        N = ncells * L
        tt = ITOTermTable(window_terms(gen, InfiniteChain(spaces), ncells), N)
        g = ITOGraph(tt, N)
        names = map(1:N) do i
            _at_site!(g, i)
            return sort([_rdesc(g, r, i) for r in eachindex(g.rrepr)])
        end
        guard = 4L + maxspan(gen)
        @test all(i -> names[i] == names[i + L], (guard + 1):(N - guard))
        @test allequal(length.(names[(guard + 1):(N - guard)]))
    end
end

@testset "bulk bonds of an unrolled sweep are literally equal" begin
    # what `_fixedpoint_cell` rests on: bulk site tensors are equal entry for entry
    S = spinvector(BSU2)
    for (name, H, L, spaces) in [
            ("Heisenberg L=1", dot(S[1], S[2]), 1, [BSU2]),
            ("J1-J2 L=1", dot(S[1], S[2]) + 0.5 * dot(S[1], S[3]), 1, [BSU2]),
            ("Heisenberg L=2", dot(S[1], S[2]) + dot(S[2], S[3]), 2, [BSU2, BSU2]),
        ]
        gen = unitcell_terms(H, L)
        R = maxspan(gen)
        ncells = 14
        N = ncells * L
        tt = ITOTermTable(window_terms(gen, InfiniteChain(spaces), ncells), N)
        Ws, secs = _irrep_sweep(tt, N, VertexCover())

        c = _fixedpoint_cell(Ws, secs, L, R)
        @test 0 < c < ncells - 1
        for cc in c:(ncells - 4)
            @test _cellsequal(Ws, L, cc)
            @test secs[(cc - 1) * L] == secs[cc * L]
        end
    end
end

@testset "the sweep does not depend on the order terms were written in" begin
    # a shuffled term list must give an identical reduced MPO, entry for entry
    S = spinvector(BSU2)
    Sp = matrixunit(BU1, U1Irrep(1 // 2), U1Irrep(-1 // 2))
    Sm = matrixunit(BU1, U1Irrep(-1 // 2), U1Irrep(1 // 2))
    Sz = spin_ops(U1Irrep).S_z
    cases = [
        (
            "SU2 J1-J2", 1, BSU2,
            [dot(S[1], S[2]), 0.5 * dot(S[1], S[3]), 0.25 * dot(S[1], S[4])],
        ),
        (
            "U1 XXZ + field", 1, BU1,
            [
                0.5 * couple(Sp[1], Sm[2]), 0.5 * couple(Sm[1], Sp[2]),
                couple(Sz[1], Sz[2]), 0.3 * Sz[1],
            ],
        ),
        (
            "SU2 dimerised", 2, BSU2,
            [dot(S[1], S[2]), 0.4 * dot(S[2], S[3]), 0.2 * dot(S[1], S[3])],
        ),
    ]
    for (name, L, V, parts) in cases
        chain = InfiniteChain(fill(V, L))
        base = _infinite_window(unitcell_terms(opsum(chain, parts)))
        for perm in (reverse(eachindex(parts)), circshift(collect(eachindex(parts)), 1))
            alt = _infinite_window(unitcell_terms(opsum(chain, parts[collect(perm)])))
            @test alt.bondsectors == base.bondsectors
            @test (alt.start, alt.finish) == (base.start, base.finish)
            @test all(j -> _entriesequal(alt.Ws[j], base.Ws[j]), 1:L)
        end
    end
end

@testset "identity-channel detection" begin
    S = spinvector(BSU2)
    H = _infinite_window(unitcell_terms(opsum(InfiniteChain([BSU2]), dot(S[1], S[2]))))
    @test _identity_channels(H.Ws, H.bondsectors) == (H.start, H.finish)

    W = H.Ws[1]
    s, d = H.start[1], H.finish[1]
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
    # textbook numbers, so a regression in `_canonicalise_bond!` is attributed here
    S = spinvector(BSU2)
    for N in (4, 6, 8)
        H = sum([dot(S[i], S[i + 1]) for i in 1:(N - 1)])
        secs = OpSum.mpo(opsum(FiniteChain(BSU2, N), H)).bondsectors
        dense = [sum(dim, sec) for sec in secs]
        @test dense[1] == 4                       # spin-1 channel + identity
        @test all(==(5), dense[2:(N - 2)])        # bulk
        @test dense[N] == 1                       # closed
    end
end
