using Test
using OpSum
using OpSum: AbstractLattice, FiniteChain, InfiniteChain, FiniteMPO, InfiniteMPO,
    windowlattice, _tolattice, _maxsite, irrep_mpo, spin, opsum, fermion_ops, couple,
    jordan_mpo_tensors, instantiate, Terms
using TensorKit
using TensorKit: sectortype
using LinearAlgebra: dot

include(joinpath(@__DIR__, "testutils.jl"))

const VSU2 = SU2Space(1 // 2 => 1)
const VSU2b = SU2Space(1 => 1)
const VU1 = Rep[U₁](0 => 1, 1 => 1)

# The lattice layer is the boundary where a latticeless operator meets physical spaces, so its
# accessors and its conversions are public surface in their own right.

@testset "FiniteChain" begin
    lat = FiniteChain([VSU2, VSU2b, VSU2])
    @test lat isa AbstractLattice
    @test length(lat) == 3
    @test !isempty(lat)
    @test eltype(lat) == typeof(VSU2)
    @test collect(lat) == [VSU2, VSU2b, VSU2]          # iterate
    @test firstindex(lat) == 1 && lastindex(lat) == 3
    @test lat[2] == VSU2b
    @test sectortype(lat) == SU2Irrep

    # the uniform convenience constructor replaces `fill(V, N)`
    @test FiniteChain(VSU2, 3) == FiniteChain(fill(VSU2, 3))
    @test FiniteChain(lat) === lat                      # idempotent
    @test occursin("FiniteChain", sprint(show, lat))

    # a finite chain has an edge: `getindex` does not wrap
    @test_throws BoundsError lat[0]
    @test_throws BoundsError lat[4]

    # equality is by type *and* spaces, so the two chain kinds never compare equal
    @test FiniteChain(VSU2, 2) != FiniteChain(VSU2b, 2)
    @test FiniteChain(VSU2, 1) != InfiniteChain([VSU2])
    @test isempty(FiniteChain(typeof(VSU2)[]))
end

@testset "InfiniteChain" begin
    lat = InfiniteChain([VSU2, VSU2b])
    @test lat isa AbstractLattice
    @test length(lat) == 2
    @test eltype(lat) == typeof(VSU2)
    @test collect(lat) == [VSU2, VSU2b]
    @test sectortype(lat) == SU2Irrep
    @test occursin("InfiniteChain", sprint(show, lat))

    # the whole point: `getindex` wraps, in both directions, so a generating term may name any site
    @test lat[1] == VSU2 && lat[2] == VSU2b
    @test lat[3] == VSU2 && lat[4] == VSU2b
    @test lat[0] == VSU2b && lat[-1] == VSU2
    @test InfiniteChain(VSU2) == InfiniteChain([VSU2])   # single-site cell

    @test_throws ArgumentError InfiniteChain(typeof(VSU2)[])

    # `windowlattice` unrolls the cell into the finite chain the infinite sweep runs on
    w = windowlattice(lat, 5)
    @test w isa FiniteChain
    @test collect(w) == [VSU2, VSU2b, VSU2, VSU2b, VSU2]
end

@testset "_maxsite distinguishes the two lattices" begin
    # this is what makes the site-range check apply to one and not the other
    @test _maxsite(FiniteChain(VSU2, 4)) == 4
    @test _maxsite(InfiniteChain([VSU2])) === nothing
end

@testset "_tolattice accepts anything naming one space per site" begin
    ref = FiniteChain(VSU2, 3)
    @test _tolattice(ref) === ref                              # passthrough
    @test _tolattice([VSU2, VSU2, VSU2]) == ref                # vector
    @test _tolattice(ntuple(_ -> VSU2, 3)) == ref              # tuple
    @test _tolattice(VSU2 for _ in 1:3) == ref                 # generator
    inf = InfiniteChain([VSU2])
    @test _tolattice(inf) === inf

    # ...and names what went wrong otherwise, rather than failing inside `collect`
    @test_throws ArgumentError _tolattice([1, 2, 3])           # wrong element type
    @test_throws ArgumentError _tolattice(3)                   # not iterable at all
    # the realistic version of that: an algorithm selector one argument too early
    @test_throws ArgumentError _tolattice(BipartiteAlgorithm())
    h = opsum(dot(spin(VSU2)[1], spin(VSU2)[2]))
    @test_throws ArgumentError irrep_mpo(h, SVDBondAlgorithm())
end

@testset "FiniteMPO / InfiniteMPO surface" begin
    S = spin(VSU2)
    N = 4
    lat = FiniteChain(VSU2, N)
    h = opsum(dot(S[i], S[i + 1]) for i in 1:(N - 1))

    mpo = irrep_mpo(h, lat)
    @test mpo isa FiniteMPO
    @test length(mpo) == N
    # destructures as the `(Ws, bondsectors)` pair, the same contract as an InfiniteMPO
    Ws, secs = mpo
    @test Ws === mpo.Ws && secs === mpo.bondsectors
    @test length(Ws) == N && length(secs) == N
    str = sprint(show, mpo)
    @test startswith(str, "FiniteMPO{")
    @test occursin("N = $N", str)

    inf = irrep_mpo(dot(S[1], S[2]), InfiniteChain([VSU2]))
    @test inf isa InfiniteMPO
    @test length(inf) == 1
    Wi, si = inf
    @test Wi === inf.Ws && si === inf.bondsectors
    istr = sprint(show, inf)
    @test startswith(istr, "InfiniteMPO{")
    @test occursin("L = 1", istr)
end

# Every entry point that takes a lattice runs the same check, so an operator can never reach a
# numerical result through spaces it was not validated against.
@testset "the lattice boundary is where operator and spaces are confronted" begin
    S = spin(VSU2)
    h = opsum(dot(S[i], S[i + 1]) for i in 1:3)
    good = FiniteChain(VSU2, 4)

    @test irrep_mpo(h, good) isa FiniteMPO

    # a term past the end of a finite chain
    @test_throws ArgumentError irrep_mpo(h, FiniteChain(VSU2, 3))
    @test_throws ArgumentError instantiate(h, FiniteChain(VSU2, 3))
    @test_throws ArgumentError islossless(h, FiniteChain(VSU2, 3))
    @test_throws ArgumentError adjoint(h, FiniteChain(VSU2, 3))
    @test_throws ArgumentError jordan_mpo_tensors(h, FiniteChain(VSU2, 3))

    # a lattice over a different symmetry
    @test_throws ArgumentError irrep_mpo(h, FiniteChain(VU1, 4))
    # a space carrying no letter of the term's charge (spin-0 has no spin-1 operator)
    @test_throws ArgumentError irrep_mpo(h, FiniteChain(SU2Space(0 => 1), 4))

    # an InfiniteChain has no site range to violate: a generating term may reach past the cell
    @test irrep_mpo(dot(S[1], S[2]), InfiniteChain([VSU2])) isa InfiniteMPO
    # but the letters are still checked against its (wrapped) spaces
    @test_throws ArgumentError irrep_mpo(dot(S[1], S[2]), InfiniteChain([SU2Space(0 => 1)]))

    # Jordan form is a finite-chain emission
    @test_throws ArgumentError jordan_mpo_tensors(dot(S[1], S[2]), InfiniteChain([VSU2]))
end

@testset "opsum is latticeless, and says so" begin
    S = spin(VSU2)
    t = dot(S[1], S[2])

    # the old lattice-binding signature is named rather than a MethodError
    @test_throws ArgumentError opsum(fill(VSU2, 4), t)
    @test_throws ArgumentError opsum(FiniteChain(VSU2, 4), t)
    @test_throws ArgumentError opsum(InfiniteChain([VSU2]), t)

    # with no terms there is no sector type to infer
    @test_throws ArgumentError opsum()
    @test_throws ArgumentError opsum(Term{SU2Irrep}[])
    # ...but an explicitly typed empty bag is fine, and establishes `I`
    @test isempty(opsum(Terms{SU2Irrep}()))

    # the postfix adjoint needs spaces and points at the replacement
    @test_throws ArgumentError t'
end

@testset "adjoint over a non-uniform and an infinite lattice" begin
    # `adjoint` only reads the spaces of the sites a term touches, so any lattice kind works
    Vf = Vect[FermionNumber](0 => 1, 1 => 1)
    F = fermion_ops(Vf)
    T = opsum(-1.0 * couple(F.cd[i], F.c[i + 1]) for i in 1:3)
    for lat in (FiniteChain(Vf, 4), fill(Vf, 4), InfiniteChain([Vf]))
        @test adjoint(T, lat) ≈ opsum(-1.0 * couple(F.cd[i + 1], F.c[i]) for i in 1:3)
    end
end

# A non-uniform chain is the case `FiniteChain(V, N)` cannot express, and nothing else covers it.
@testset "non-uniform FiniteChain compresses" begin
    Sa, Sb = spin(VSU2), spin(VSU2b)
    lat = FiniteChain([VSU2, VSU2b, VSU2, VSU2b])
    h = opsum(
        dot((isodd(i) ? Sa : Sb)[i], (isodd(i + 1) ? Sa : Sb)[i + 1]) for i in 1:3
    )
    @test islossless(h, lat)
    Ws, secs = irrep_mpo(h, lat)
    @test length(secs) == 4
    @test secs[4] == [SU2Irrep(0)]
    # the alternating spaces are what the assembled tensors have to honour
    Ts = irrep_mpo_tensors(irrep_mpo(h, lat), lat)
    @test space(Ts[1], 2) == VSU2
    @test space(Ts[2], 2) == VSU2b
end
