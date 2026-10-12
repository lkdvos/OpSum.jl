using Test
using OpSum
using OpSum: AbstractLattice, FiniteChain, InfiniteChain,
    windowlattice, _tolattice, _maxsite, mpo, opsum, couple,
    jordan_mpo_tensors, instantiate, Terms
using TensorKit
using TensorKit: sectortype
using LinearAlgebra: dot

include(joinpath(@__DIR__, "testutils.jl"))

const VSU2 = SU2Space(1 // 2 => 1)
const VSU2b = SU2Space(1 => 1)
const VU1 = Rep[U₁](0 => 1, 1 => 1)

# Every site of a lattice is a `Basis`; the complete matrix-unit basis stands in for the bare space.
const BSU2 = unitbasis(VSU2)
const BSU2b = unitbasis(VSU2b)
const BU1 = unitbasis(VU1)
const BSU2_0 = unitbasis(SU2Space(0 => 1))

# The lattice layer is the boundary where a latticeless operator meets physical spaces, so its
# accessors and its conversions are public surface in their own right.

@testset "FiniteChain" begin
    lat = FiniteChain([BSU2, BSU2b, BSU2])
    @test lat isa AbstractLattice
    @test length(lat) == 3
    @test !isempty(lat)
    @test eltype(lat) == typeof(BSU2)
    @test collect(lat) == [BSU2, BSU2b, BSU2]          # iterate
    @test firstindex(lat) == 1 && lastindex(lat) == 3
    @test lat[2] == BSU2b
    @test sectortype(lat) == SU2Irrep

    # the uniform convenience constructor replaces `fill(V, N)`
    @test FiniteChain(BSU2, 3) == FiniteChain(fill(BSU2, 3))
    @test FiniteChain(lat) === lat                      # idempotent
    @test occursin("FiniteChain", sprint(show, lat))

    # a lattice of bare spaces is gone: every site is a `Basis`
    @test_throws ArgumentError FiniteChain(VSU2, 3)
    @test_throws ArgumentError FiniteChain([VSU2, VSU2b])
    @test_throws ArgumentError InfiniteChain([VSU2])

    # a finite chain has an edge: `getindex` does not wrap
    @test_throws BoundsError lat[0]
    @test_throws BoundsError lat[4]

    # equality is by type *and* spaces, so the two chain kinds never compare equal
    @test FiniteChain(BSU2, 2) != FiniteChain(BSU2b, 2)
    @test FiniteChain(BSU2, 1) != InfiniteChain([BSU2])
    @test isempty(FiniteChain(typeof(BSU2)[]))
end

@testset "InfiniteChain" begin
    lat = InfiniteChain([BSU2, BSU2b])
    @test lat isa AbstractLattice
    @test length(lat) == 2
    @test eltype(lat) == typeof(BSU2)
    @test collect(lat) == [BSU2, BSU2b]
    @test sectortype(lat) == SU2Irrep
    @test occursin("InfiniteChain", sprint(show, lat))

    # the whole point: `getindex` wraps, in both directions, so a generating term may name any site
    @test lat[1] == BSU2 && lat[2] == BSU2b
    @test lat[3] == BSU2 && lat[4] == BSU2b
    @test lat[0] == BSU2b && lat[-1] == BSU2
    @test InfiniteChain(BSU2) == InfiniteChain([BSU2])   # single-site cell

    @test_throws ArgumentError InfiniteChain(typeof(BSU2)[])

    # `windowlattice` unrolls the cell into the finite chain the infinite sweep runs on
    w = windowlattice(lat, 5)
    @test w isa FiniteChain
    @test collect(w) == [BSU2, BSU2b, BSU2, BSU2b, BSU2]
end

@testset "_maxsite distinguishes the two lattices" begin
    # this is what makes the site-range check apply to one and not the other
    @test _maxsite(FiniteChain(BSU2, 4)) == 4
    @test _maxsite(InfiniteChain([BSU2])) === nothing
end

@testset "_tolattice accepts anything naming one space per site" begin
    ref = FiniteChain(BSU2, 3)
    @test _tolattice(ref) === ref                              # passthrough
    @test _tolattice([BSU2, BSU2, BSU2]) == ref                # vector
    @test _tolattice(ntuple(_ -> BSU2, 3)) == ref              # tuple
    @test _tolattice(BSU2 for _ in 1:3) == ref                 # generator
    inf = InfiniteChain([BSU2])
    @test _tolattice(inf) === inf

    # ...and names what went wrong otherwise, rather than failing inside `collect`
    @test_throws ArgumentError _tolattice([1, 2, 3])           # wrong element type
    @test_throws ArgumentError _tolattice(3)                   # not iterable at all
    # the realistic version of that: an algorithm selector one argument too early
    @test_throws ArgumentError _tolattice(BipartiteAlgorithm())
    h = opsum(dot(spinvector(BSU2)[1], spinvector(BSU2)[2]))
    @test_throws ArgumentError mpo(h, SVDBondAlgorithm())
end

@testset "SymbolicMPO surface" begin
    S = spinvector(BSU2)
    N = 4
    lat = FiniteChain(BSU2, N)
    h = opsum(dot(S[i], S[i + 1]) for i in 1:(N - 1))

    W = mpo(opsum(lat, h))
    @test W isa SymbolicMPO{SU2Irrep, <:FiniteChain}
    @test length(W) == N && W.lattice == lat
    # iterates over its sites; `reduced` is the `(Ws, bondsectors)` pair
    @test collect(W) == W.Ws
    Ws, secs = reduced(W)
    @test Ws === W.Ws && secs === W.bondsectors
    @test length(Ws) == N && length(secs) == N
    str = sprint(show, W)
    @test startswith(str, "SymbolicMPO{")
    @test occursin("N = $N", str)

    inf = mpo(opsum(InfiniteChain([BSU2]), dot(S[1], S[2])))
    @test inf isa SymbolicMPO{SU2Irrep, <:InfiniteChain}
    @test length(inf) == 1
    @test occursin("InfiniteChain, N = 1", sprint(show, inf))
end

# Terms are checked as they enter the container, so an operator can never reach a numerical result
# through spaces it was not validated against, and the error points at the line that added it.
@testset "the lattice boundary is where operator and spaces are confronted" begin
    S = spinvector(BSU2)
    h = opsum(dot(S[i], S[i + 1]) for i in 1:3)
    good = FiniteChain(BSU2, 4)

    @test mpo(opsum(good, h)) isa SymbolicMPO

    # a term past the end of a finite chain
    @test_throws ArgumentError opsum(FiniteChain(BSU2, 3), h)
    @test_throws ArgumentError opsum!(OperatorSum(FiniteChain(BSU2, 3)), h)
    @test_throws ArgumentError opsum(FiniteChain(BSU2, 3)) + h

    # a lattice over a different symmetry
    @test_throws ArgumentError opsum(FiniteChain(BU1, 4), h)
    # a space carrying no letter of the term's charge (spin-0 has no spin-1 operator)
    @test_throws ArgumentError opsum(FiniteChain(BSU2_0, 4), h)

    # an InfiniteChain has no site range to violate: a generating term may reach past the cell
    @test mpo(opsum(InfiniteChain([BSU2]), dot(S[1], S[2]))) isa SymbolicMPO
    # but the letters are still checked against its (wrapped) spaces
    @test_throws ArgumentError opsum(InfiniteChain([BSU2_0]), dot(S[1], S[2]))

    # the unit cell has a Jordan form too
    @test length(jordan_mpo_tensors(opsum(InfiniteChain([BSU2]), dot(S[1], S[2])))) == 1
end

@testset "opsum without a lattice is latticeless; with one it is an OperatorSum" begin
    S = spinvector(BSU2)
    t = dot(S[1], S[2])

    @test opsum(t) isa Terms
    # a lattice (or a vector of spaces) as the first argument binds it
    @test opsum(fill(BSU2, 4), t) isa OperatorSum
    @test opsum(FiniteChain(BSU2, 4), t) isa OperatorSum
    @test opsum(InfiniteChain([BSU2]), t) isa OperatorSum
    @test opsum(fill(BSU2, 4), t) ≈ opsum(FiniteChain(BSU2, 4), t)

    # with no terms there is no sector type to infer
    @test_throws ArgumentError opsum()
    @test_throws ArgumentError opsum(Term{SU2Irrep}[])
    # ...but an explicitly typed empty bag is fine, and establishes `I`
    @test isempty(opsum(Terms{SU2Irrep}()))
    # ...and a lattice names the sector on its own
    @test isempty(opsum(FiniteChain(BSU2, 4)))

    # the postfix adjoint of a bare bag needs spaces and points at the replacement
    @test_throws ArgumentError t'
end

@testset "adjoint over a non-uniform and an infinite lattice" begin
    # `adjoint` only reads the spaces of the sites a term touches, so any lattice kind works
    Vf = Vect[FermionNumber](0 => 1, 1 => 1)
    Bf = unitbasis(Vf)
    F = fermionmode(Bf, FermionNumber(0), FermionNumber(1))
    T = opsum(-1.0 * couple(F.cd[i], F.c[i + 1]) for i in 1:3)
    for lat in (FiniteChain(Bf, 4), fill(Bf, 4), InfiniteChain([Bf]))
        @test opsum(lat, T)' ≈ opsum(lat, -1.0 * couple(F.cd[i + 1], F.c[i]) for i in 1:3)
    end
end

# A non-uniform chain is the case `FiniteChain(V, N)` cannot express, and nothing else covers it.
@testset "non-uniform FiniteChain compresses" begin
    Sa, Sb = spinvector(BSU2), spinvector(BSU2b)
    lat = FiniteChain([BSU2, BSU2b, BSU2, BSU2b])
    h = opsum(
        dot((isodd(i) ? Sa : Sb)[i], (isodd(i + 1) ? Sa : Sb)[i + 1]) for i in 1:3
    )
    H = opsum(lat, h)
    @test islossless(H)
    Ws, secs = reduced(mpo(H))
    @test length(secs) == 4
    @test secs[4] == [SU2Irrep(0)]
    # the alternating spaces are what the assembled tensors have to honour
    Ts = irrep_mpo_tensors(mpo(H))
    @test space(Ts[1], 2) == VSU2
    @test space(Ts[2], 2) == VSU2b
end
