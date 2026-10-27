using Test
using TensorKit
using LinearAlgebra: dot
using OpSum
using OpSum: couple, unitcell_terms, window_terms, maxspan,
    TermTable, SweepGraph, rdesc, infinite_window, at_site!

# Translation covariance of the sweep: live suffix-class names repeat with the unit cell, and the result
# does not depend on the order terms were written in.

const VSU2 = SU2Space(1 // 2 => 1)
const VU1 = U1Space(-1 // 2 => 1, 1 // 2 => 1)

include(joinpath(@__DIR__, "..", "testutils.jl"))   # unitbasis, matrixunit, spinvector
const BSU2 = unitbasis(VSU2)
const BU1 = spin_basis(U1Irrep)   # stock basis: Sᶻ is one atomic letter

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
        tt = TermTable(window_terms(gen, InfiniteChain(spaces), ncells))
        g = SweepGraph(tt, N)
        names = map(1:N) do i
            at_site!(g, i)
            return sort([rdesc(g, r, i) for r in eachindex(g.right.repr)])
        end
        guard = 4L + maxspan(gen)
        @test all(i -> names[i] == names[i + L], (guard + 1):(N - guard))
        @test allequal(length.(names[(guard + 1):(N - guard)]))
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
        base = infinite_window(unitcell_terms(opsum(chain, parts)))
        for perm in (reverse(eachindex(parts)), circshift(collect(eachindex(parts)), 1))
            alt = infinite_window(unitcell_terms(opsum(chain, parts[collect(perm)])))
            @test alt.bondsectors == base.bondsectors
            @test (alt.start, alt.finish) == (base.start, base.finish)
            @test alt.Ws == base.Ws
        end
    end
end

@testset "the cell's identity channels" begin
    S = spinvector(BSU2)
    H = infinite_window(unitcell_terms(opsum(InfiniteChain([BSU2]), dot(S[1], S[2]))))
    W = H.Ws[1]
    s, d = H.start[1], H.finish[1]
    @test H.bondsectors[1][s] == H.bondsectors[1][d] == unit(SU2Irrep)
    @test s != d
    # nothing enters the start channel but itself, nothing leaves the finish channel but itself
    entries = collect(OpSum.storedpairs(W))
    @test count(p -> Tuple(first(p))[2] == s, entries) == 1
    @test count(p -> Tuple(first(p))[1] == d, entries) == 1
    @test isone(W[s, s]) && isone(W[d, d])
end
