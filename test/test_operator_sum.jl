using Test
using OpSum
using OpSum: OperatorSum, opsum!, opsum, irrep_mpo, irrep_mpo_tensors, jordan_mpo_tensors,
    instantiate, islossless, mpo_tensormap, chain_terms, expterm, Terms, FiniteChain, InfiniteChain,
    FiniteMPO, InfiniteMPO, couple, spin, spin_ops, fermion_ops
using TensorKit
using LinearAlgebra: dot

include(joinpath(@__DIR__, "testutils.jl"))   # physmatrix, densedim

const VSU2 = SU2Space(1 // 2 => 1)
const VSU2_0 = SU2Space(0 => 1)
const VU1 = Rep[U₁](0 => 1, 1 => 1)
const VF = Vect[FermionNumber](0 => 1, 1 => 1)
const S = spin(VSU2)
const b = dot(S, S)

# the message of the ArgumentError `f()` throws
function argerr(f)
    try
        f()
    catch e
        e isa ArgumentError && return e.msg
        rethrow()
    end
    return nothing
end

@testset "the four cells share one container and one signature" begin
    fin, inf = FiniteChain(VSU2, 8), InfiniteChain(VSU2)
    cells = [
        (opsum(fin, (b[i] for i in 1:7)), FiniteMPO, [4, 5, 5, 5, 5, 5, 4, 1]),
        (opsum(inf, b[1]), InfiniteMPO, [5]),
        (opsum(fin, b[1], expterm(b[1]; decay = 0.4)), FiniteMPO, [7, 5, 5, 5, 5, 5, 4, 1]),
        (opsum(inf, b[1], expterm(b[1]; decay = 0.4)), InfiniteMPO, [8]),
    ]
    for (H, T, D) in cells
        @test H isa OperatorSum
        mpo = irrep_mpo(H)
        @test mpo isa T
        @test [densedim(mpo.bondsectors, k) for k in eachindex(mpo.bondsectors)] == D
        # islossless is a finite-chain check; an infinite chain is refused
        H.lattice isa FiniteChain ? (@test islossless(H)) : @test_throws ArgumentError islossless(H)
    end

    # the dense oracle of a decaying model is the geometric sum truncated to the chain
    Hs = opsum(FiniteChain(VSU2, 4), b[1], expterm(b[1]; decay = 0.4))
    @test instantiate(Hs) ≈ instantiate(opsum(FiniteChain(VSU2, 4), chain_terms(Hs)))
    @test mpo_tensormap(irrep_mpo_tensors(irrep_mpo(Hs), Hs.lattice)) ≈ instantiate(Hs)

    # SVD and Jordan form exist on a plain finite chain only
    @test irrep_mpo(cells[1][1], SVDBondAlgorithm(truncrank(6))) isa FiniteMPO
    @test length(jordan_mpo_tensors(cells[1][1])) == 8
    for i in 2:4
        @test_throws ArgumentError jordan_mpo_tensors(cells[i][1])
        @test_throws ArgumentError irrep_mpo(cells[i][1], SVDBondAlgorithm(truncrank(6)))
    end
    @test_throws ArgumentError instantiate(cells[2][1])
end

@testset "opsum!, opsum(lat, …) and += agree" begin
    N = 6
    lat = FiniteChain(VSU2, N)
    H1 = opsum(lat, (b[i] for i in 1:(N - 1)))
    H2 = OperatorSum(lat)
    @test opsum!(H2, (b[i] for i in 1:(N - 1))) === H2
    H3 = OperatorSum(lat)
    for i in 1:(N - 1)
        H3 += b[i]
    end
    H4 = opsum!(OperatorSum(fill(VSU2, N)), [b[i] for i in 1:(N - 1)])   # a vector of spaces is a lattice
    @test H4.lattice == lat
    @test H1 ≈ H2 && H1 ≈ H3 && H1 ≈ H4

    # `+` copies, `opsum!` mutates; coincident terms sum
    H = OperatorSum(lat)
    G = H + b[1]
    @test isempty(H) && length(G) == 1
    opsum!(H, b[1], b[1])
    @test H ≈ 2 * G && (2H) / 2 ≈ H && isempty((H - H).terms)

    # a failed insertion leaves the container untouched; bad arguments are named
    H = opsum(lat, b[1])
    @test_throws ArgumentError opsum!(H, b[2], b[N])      # b[N] reaches site N + 1
    @test length(H) == 1
    @test_throws ArgumentError opsum!(H, S)               # unplaced SiteOperator
    @test_throws ArgumentError opsum!(H, b)               # unplaced LocalOperator
    @test_throws ArgumentError opsum!(H, 1.0)
end

@testset "H + H' on hopping chains" begin
    # U(1) spin-½, against a hand-built hermitian operator
    N = 4
    lat = FiniteChain(VU1, N)
    U = spin_ops(VU1, U1Irrep(1), U1Irrep(0))
    T = opsum(lat, couple(U.Sp[i], U.Sm[i + 1]) for i in 1:(N - 1))
    H = T + T'
    @test islossless(H)
    Sp, Sm, id2 = [0 0; 1 0], [0 1; 0 0], [1 0; 0 1]
    site(ops) = foldl(kron, [get(ops, i, id2) for i in N:-1:1])
    Hhand = sum(site(Dict(i => Sp, i + 1 => Sm)) + site(Dict(i => Sm, i + 1 => Sp)) for i in 1:(N - 1))
    @test physmatrix(instantiate(H), N, 2) ≈ Hhand

    # fermions: the adjoint supplies the anticommutation sign; infinite chains work too
    F = fermion_ops(VF)
    hops(i) = -1.0 * (couple(F.cd[i], F.c[i + 1]) + couple(F.cd[i + 1], F.c[i]))
    Tf = opsum(FiniteChain(VF, 5), -1.0 * couple(F.cd[i], F.c[i + 1]) for i in 1:4)
    @test Tf + Tf' ≈ opsum(FiniteChain(VF, 5), hops(i) for i in 1:4)
    ilat = InfiniteChain(VF)
    Ti = opsum(ilat, -1.0 * couple(F.cd[1], F.c[2]))
    @test irrep_mpo(Ti + Ti').bondsectors == irrep_mpo(opsum(ilat, hops(1))).bondsectors
    @test_throws ArgumentError couple(F.cd[1], F.c[2])'                       # bare bag: no h'
    @test_throws ArgumentError opsum(FiniteChain(VSU2, 2), S[1])'             # charged term
    @test_throws ArgumentError opsum(lat, expterm(couple(U.Sp[1], U.Sm[2]); decay = 0.3))'
end

@testset "checks when terms enter" begin
    lat = FiniteChain(VSU2, 4)
    Su1 = spin_ops(VU1, U1Irrep(1), U1Irrep(0))
    @test occursin("sector type", argerr(() -> opsum(lat, couple(Su1.Sz[1], Su1.Sz[2]))))
    @test occursin(
        "does not exist on site 2", argerr(() -> opsum(FiniteChain([VSU2, VSU2_0, VSU2]), b[1]))
    )
    @test occursin("outside the lattice `1:4`", argerr(() -> opsum(lat, b[4])))
    @test length(opsum(InfiniteChain(VSU2), b[4])) == 1                       # an infinite chain wraps

    # an infinite chain holds a generating set: neutral, no K = 0, letters valid on the cell
    inf = InfiniteChain(VSU2)
    @test occursin("charge-neutral", argerr(() -> opsum(inf, S[1])))
    @test occursin("K = 0", argerr(() -> opsum(inf, one(Terms{SU2Irrep}))))
    @test_throws ArgumentError opsum(inf, expterm(couple(S[1], S[2]; to = SU2Irrep(1)); decay = 0.5))
    @test_throws ArgumentError opsum(InfiniteChain(VSU2_0), expterm(b[1]; decay = 0.5))
    # two representatives of one translation class are only visible across the set
    @test occursin("translates of each other", argerr(() -> irrep_mpo(opsum(inf, b[1], b[2]))))
    @test irrep_mpo(opsum(inf, b[1], b[1])) isa InfiniteMPO                   # a term twice is one term

    @test_throws ArgumentError instantiate(OperatorSum(lat))
    # the old `(h, lat)` forms point at the container
    @test occursin("opsum(lat", argerr(() -> irrep_mpo(b[1], lat)))
    @test_throws MethodError islossless(b[1], lat)
end

@testset "combining operators" begin
    lat = FiniteChain(VSU2, 6)
    a, c = opsum(lat, b[1]), opsum(lat, b[2])
    @test a + c ≈ opsum(lat, b[1], b[2])
    @test a - c ≈ opsum(lat, b[1], -b[2])
    for other in (FiniteChain(VSU2, 5), InfiniteChain(VSU2), FiniteChain(VSU2_0, 6))
        @test_throws ArgumentError a + OperatorSum(other)
    end
    @test !(a ≈ opsum(FiniteChain(VSU2, 5), b[1]))
    # channels travel with the container
    e = expterm(b[1]; decay = 0.3)
    @test length((a + e).channels.channels) == 1 && (a + e) ≈ (a + opsum(lat, e))
end
