using Test
using OpSum
using OpSum: AbstractLattice, FiniteChain, InfiniteChain,
    mpo, opsum, couple,
    jordan_mpo_tensors, instantiate, Terms
using TensorKit
using TensorKit: sectortype
using LinearAlgebra: dot

include(joinpath(@__DIR__, "testutils.jl"))

const VSU2 = SU2Space(1 // 2 => 1)
const VSU2b = SU2Space(1 => 1)
const VU1 = Rep[U₁](0 => 1, 1 => 1)

const BSU2 = unitbasis(VSU2)
const BSU2b = unitbasis(VSU2b)
const BU1 = unitbasis(VU1)
const BSU2_0 = unitbasis(SU2Space(0 => 1))

@testset "FiniteChain" begin
    lat = FiniteChain([BSU2, BSU2b, BSU2])
    @test lat isa AbstractLattice
    @test length(lat) == 3
    @test eltype(lat) == typeof(BSU2)
    @test collect(lat) == [BSU2, BSU2b, BSU2]          # iterate
    @test lat[2] == BSU2b
    @test sectortype(lat) == SU2Irrep

    @test FiniteChain(BSU2, 3) == FiniteChain(fill(BSU2, 3))
    @test FiniteChain(lat) === lat                      # idempotent

    # every site is a `Basis`; bare spaces are rejected
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

    # the whole point: `getindex` wraps, in both directions, so a generating term may name any site
    @test lat[1] == BSU2 && lat[2] == BSU2b
    @test lat[3] == BSU2 && lat[4] == BSU2b
    @test lat[0] == BSU2b && lat[-1] == BSU2

    @test_throws ArgumentError InfiniteChain(typeof(BSU2)[])
end

@testset "lattice constructors take one Basis per site" begin
    ref = FiniteChain(BSU2, 3)
    @test FiniteChain([BSU2, BSU2, BSU2]) == ref                # vector of bases
    @test OperatorSum([BSU2, BSU2, BSU2]).lattice == ref
    @test opsum([BSU2, BSU2, BSU2]).lattice == ref
    @test InfiniteChain([BSU2, BSU2b]) isa InfiniteChain

    @test_throws ArgumentError FiniteChain([1, 2, 3])           # wrong element type
    @test_throws ArgumentError InfiniteChain([VSU2])            # a bare space is not an alphabet
    @test_throws ArgumentError FiniteChain(VSU2, 3)
    @test_throws ArgumentError FiniteChain([BSU2, BU1])         # two symmetries
    h = Terms(dot(spinvector(BSU2)[1], spinvector(BSU2)[2]))
    @test_throws ArgumentError mpo(h, IndependentSVD())
end

@testset "SymbolicMPO surface" begin
    S = spinvector(BSU2)
    N = 4
    lat = FiniteChain(BSU2, N)
    h = Terms(dot(S[i], S[i + 1]) for i in 1:(N - 1))

    W = mpo(opsum(lat, h))
    @test W isa SymbolicMPO{SU2Irrep, <:FiniteChain}
    @test length(W) == N && W.lattice == lat
    # iterates over its sites; `reduced` is the `(Ws, bondsectors)` pair
    @test collect(W) == W.Ws
    Ws, secs = reduced(W)
    @test Ws === W.Ws && secs === W.bondsectors
    @test length(Ws) == N && length(secs) == N
    @test occursin("N = $N", sprint(show, W))

    inf = mpo(opsum(InfiniteChain([BSU2]), dot(S[1], S[2])))
    @test inf isa SymbolicMPO{SU2Irrep, <:InfiniteChain}
    @test length(inf) == 1
end

@testset "the lattice boundary is where operator and spaces are confronted" begin
    S = spinvector(BSU2)
    h = Terms(dot(S[i], S[i + 1]) for i in 1:3)
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
end

@testset "Terms collects latticeless; opsum with a lattice is an OperatorSum" begin
    S = spinvector(BSU2)
    t = dot(S[1], S[2])

    @test Terms(t) isa Terms
    # a lattice (or a vector of bases) as the first argument binds it
    @test opsum(fill(BSU2, 4), t) isa OperatorSum
    @test opsum(FiniteChain(BSU2, 4), t) isa OperatorSum
    @test opsum(InfiniteChain([BSU2]), t) isa OperatorSum
    @test opsum(fill(BSU2, 4), t) ≈ opsum(FiniteChain(BSU2, 4), t)

    # with no terms there is no sector type to infer
    @test_throws ArgumentError Terms()
    @test_throws ArgumentError Terms(Any[])
    @test_throws ArgumentError Terms(Any[], Any[])
    @test isempty(Terms{SU2Irrep}()) && isempty(Terms{SU2Irrep}(Any[])) && isempty(Terms(Term{SU2Irrep}[]))
    @test isempty(Terms(Terms{SU2Irrep}())) && isempty(opsum(FiniteChain(BSU2, 4)))
    @test_throws MethodError opsum(t)                       # `opsum` always has a lattice
    @test Terms{SU2Irrep}(t, [t]) ≈ 2 * t && Terms(Term{SU2Irrep}[only(t)]) ≈ t
    @test_throws ArgumentError Terms{U1Irrep}(t)            # a term of another sector

    # the postfix adjoint of a bare bag needs spaces and points at the replacement
    @test_throws ArgumentError t'
end

@testset "adjoint over a non-uniform and an infinite lattice" begin
    Vf = Vect[FermionNumber](0 => 1, 1 => 1)
    Bf = unitbasis(Vf)
    F = fermionmode(Bf, FermionNumber(0), FermionNumber(1))
    T = Terms(-1.0 * couple(F.cd[i], F.c[i + 1]) for i in 1:3)
    for lat in (FiniteChain(Bf, 4), fill(Bf, 4), InfiniteChain([Bf]))
        @test opsum(lat, T)' ≈ opsum(lat, -1.0 * couple(F.cd[i + 1], F.c[i]) for i in 1:3)
    end
end

# A non-uniform chain is the case `FiniteChain(V, N)` cannot express, and nothing else covers it.
@testset "non-uniform FiniteChain compresses" begin
    Sa, Sb = spinvector(BSU2), spinvector(BSU2b)
    lat = FiniteChain([BSU2, BSU2b, BSU2, BSU2b])
    h = Terms(
        dot((isodd(i) ? Sa : Sb)[i], (isodd(i + 1) ? Sa : Sb)[i + 1]) for i in 1:3
    )
    H = opsum(lat, h)
    @test islossless(H)
    Ws, secs = reduced(mpo(H))
    @test length(secs) == 4
    @test secs[4] == [SU2Irrep(0)]
    # the alternating spaces are what the assembled tensors have to honour
    Ts = mpo_tensors(mpo(H))
    @test space(Ts[1], 2) == VSU2
    @test space(Ts[2], 2) == VSU2b
end
