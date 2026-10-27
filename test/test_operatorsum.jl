using Test
using OpSum
using OpSum: OperatorSum, opsum!, opsum, mpo, mpo_tensors, jordan_mpo_tensors,
    instantiate, islossless, mpo_tensormap, chain_terms, expterm, Terms, FiniteChain, InfiniteChain, couple
using TensorKit
using LinearAlgebra: dot

include(joinpath(@__DIR__, "testutils.jl"))   # physmatrix, densedim

# Spin-½ SU(2): `b` is S⃗ᵢ·S⃗ⱼ, `S` the charged spin vector (never a Hamiltonian term).
const BSU2 = spin_basis(SU2Irrep)
const SU2OPS = spin_ops(SU2Irrep)
const S = spinvector(BSU2)
const b = SU2OPS.S_exchange
# a spin-0 site carries no operator but the identity
const BSU2_0 = unitbasis(SU2Space(0 => 1))

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
    fin, inf = FiniteChain(BSU2, 4), InfiniteChain(BSU2)
    cells = [
        (opsum(fin, (b[i] for i in 1:3)), FiniteChain, [4, 5, 4, 1]),
        (opsum(inf, b[1]), InfiniteChain, [5]),
        (opsum(fin, b[1], expterm(b[1]; decay = 0.4)), FiniteChain, [7, 5, 4, 1]),
        (opsum(inf, b[1], expterm(b[1]; decay = 0.4)), InfiniteChain, [8]),
    ]
    for (H, L, D) in cells
        @test H isa OperatorSum
        W = mpo(H)
        @test W isa SymbolicMPO{SU2Irrep, <:L}
        @test [densedim(W.bondsectors, k) for k in eachindex(W.bondsectors)] == D
        H.lattice isa FiniteChain ? (@test islossless(H)) : @test_throws ArgumentError islossless(H)
    end

    Hs = opsum(FiniteChain(BSU2, 3), b[1], expterm(b[1]; decay = 0.4))
    @test instantiate(Hs) ≈ instantiate(opsum(FiniteChain(BSU2, 3), chain_terms(Hs)))
    @test mpo_tensormap(mpo_tensors(mpo(Hs))) ≈ instantiate(Hs)

    # SVD exists on a plain finite chain only; Jordan form everywhere
    @test mpo(cells[1][1], IndependentSVD(truncrank(6))) isa SymbolicMPO
    @test length(jordan_mpo_tensors(cells[1][1])) == 4
    @test length(jordan_mpo_tensors(cells[2][1])) == 1
    @test length(jordan_mpo_tensors(cells[3][1])) == 4
    @test length(jordan_mpo_tensors(cells[4][1])) == 1
    for i in 2:4
        @test_throws ArgumentError mpo(cells[i][1], IndependentSVD(truncrank(6)))
    end
    @test_throws ArgumentError instantiate(cells[2][1])
end

@testset "opsum!, opsum(lat, …) and += agree" begin
    N = 6
    lat = FiniteChain(BSU2, N)
    H1 = opsum(lat, (b[i] for i in 1:(N - 1)))
    H2 = OperatorSum(lat)
    @test opsum!(H2, (b[i] for i in 1:(N - 1))) === H2
    H3 = OperatorSum(lat)
    for i in 1:(N - 1)
        H3 += b[i]
    end
    H4 = opsum!(OperatorSum(fill(BSU2, N)), [b[i] for i in 1:(N - 1)])   # a vector of bases is a lattice
    @test H4.lattice == lat
    @test H1 ≈ H2 && H1 ≈ H3 && H1 ≈ H4

    H = OperatorSum(lat)
    G = H + b[1]
    @test isempty(H) && length(G) == 1
    opsum!(H, b[1], b[1])
    @test H ≈ 2 * G && (2H) / 2 ≈ H && isempty((H - H).terms)

    # a failed insertion leaves the container untouched
    H = opsum(lat, b[1])
    @test_throws ArgumentError opsum!(H, b[2], b[N])      # b[N] reaches site N + 1
    @test length(H) == 1
    @test_throws ArgumentError opsum!(H, S)               # unplaced SiteOperator
    @test_throws ArgumentError opsum!(H, 1.0)
end

@testset "H + H' on hopping chains" begin
    N = 3
    lat = FiniteChain(spin_basis(U1Irrep), N)
    U = spin_ops(U1Irrep)
    T = opsum(lat, U.S_plus_S_min[i, i + 1] for i in 1:(N - 1))
    H = T + T'
    @test islossless(H)
    Sp, Sm, id2 = [0 0; 1 0], [0 1; 0 0], [1 0; 0 1]
    site(ops) = foldl(kron, [get(ops, i, id2) for i in N:-1:1])
    Hhand = sum(site(Dict(i => Sp, i + 1 => Sm)) + site(Dict(i => Sm, i + 1 => Sp)) for i in 1:(N - 1))
    @test physmatrix(instantiate(H), N, 2) ≈ Hhand

    # fermions: the adjoint supplies the anticommutation sign
    F = fermion_ops()
    BF = fermion_basis()
    cdc = F.f_plus_f_min
    hops(i) = -1.0 * (cdc[i, i + 1] + cdc[i + 1, i])
    Tf = opsum(FiniteChain(BF, 3), -1.0 * cdc[i, i + 1] for i in 1:2)
    @test Tf + Tf' ≈ opsum(FiniteChain(BF, 3), hops(i) for i in 1:2)
    ilat = InfiniteChain(BF)
    Ti = opsum(ilat, -1.0 * cdc[1, 2])
    @test mpo(Ti + Ti').bondsectors == mpo(opsum(ilat, hops(1))).bondsectors
    @test_throws ArgumentError cdc[1, 2]'                       # bare bag: no h'
    @test_throws ArgumentError opsum(FiniteChain(BSU2, 2), S[1])'             # charged term
    @test_throws ArgumentError opsum(lat, expterm(U.S_plus_S_min[1, 2]; decay = 0.3))'
end

@testset "checks when terms enter" begin
    lat = FiniteChain(BSU2, 4)
    Su1 = spin_ops(U1Irrep)
    @test occursin("sector type", argerr(() -> opsum(lat, Su1.S_z_S_z[1, 2])))
    @test occursin(
        "does not exist on site 2", argerr(() -> opsum(FiniteChain([BSU2, BSU2_0, BSU2]), b[1]))
    )
    @test occursin("outside the lattice `1:4`", argerr(() -> opsum(lat, b[4])))
    @test length(opsum(InfiniteChain(BSU2), b[4])) == 1                       # an infinite chain wraps

    inf = InfiniteChain(BSU2)
    @test occursin("charge-neutral", argerr(() -> opsum(inf, S[1])))
    @test occursin("K = 0", argerr(() -> opsum(inf, one(Terms{SU2Irrep}))))
    @test_throws ArgumentError opsum(inf, expterm(couple(S[1], S[2]; to = SU2Irrep(1)); decay = 0.5))
    @test_throws ArgumentError opsum(InfiniteChain(BSU2_0), expterm(b[1]; decay = 0.5))
    @test occursin("translates of each other", argerr(() -> mpo(opsum(inf, b[1], b[2]))))
    @test mpo(opsum(inf, b[1], b[1])) isa SymbolicMPO                   # a term twice is one term

    @test_throws ArgumentError instantiate(OperatorSum(lat))
    @test occursin("opsum(lat", argerr(() -> mpo(b[1], lat)))
    @test_throws MethodError islossless(b[1], lat)
end

@testset "combining operators" begin
    lat = FiniteChain(BSU2, 6)
    a, c = opsum(lat, b[1]), opsum(lat, b[2])
    @test a + c ≈ opsum(lat, b[1], b[2])
    @test a - c ≈ opsum(lat, b[1], -b[2])
    for other in (FiniteChain(BSU2, 5), InfiniteChain(BSU2), FiniteChain(BSU2_0, 6))
        @test_throws ArgumentError a + OperatorSum(other)
    end
    @test !(a ≈ opsum(FiniteChain(BSU2, 5), b[1]))
    e = expterm(b[1]; decay = 0.3)
    @test length((a + e).channels.channels) == 1 && (a + e) ≈ (a + opsum(lat, e))
end

@testset "exact == and hash" begin
    lat = FiniteChain(BSU2, 6)
    e = expterm(b[1]; decay = 0.3)
    H = opsum(lat, b[1], 2 * b[2], e)
    @test H == copy(H) && hash(H) == hash(copy(H))
    @test H != 2H && H != H + b[3] && H != H + e
    @test H != opsum(FiniteChain(BSU2, 5), b[1], 2 * b[2], e)      # another lattice
    @test H != opsum(lat, b[1], 2 * b[2])                          # the channels count
    # the order terms and channels were added in does not matter
    G = opsum(lat, e, 2 * b[2], b[1])
    @test G == H && hash(G) == hash(H)
    e2 = expterm(b[2]; decay = -0.5)
    @test opsum(lat, e, e2) == opsum(lat, e2, e) && hash(opsum(lat, e, e2)) == hash(opsum(lat, e2, e))
    @test (H - H) == OperatorSum(lat) && hash(H - H) == hash(OperatorSum(lat))
    @test length(Set([H, copy(H), G])) == 1
end

@testset "Terms(…) and opsum accept and reject the same inputs" begin
    lat = FiniteChain(BSU2, 4)
    # accepted by both: terms, bags, nested iterables
    @test Terms(b[1], [b[2], (b[3],)]) ≈ opsum(lat, b[1], [b[2], (b[3],)]).terms
    # rejected by both, with the same ArgumentError: numbers, unplaced operators, OperatorSums
    for bad in (1.0, 2, S, one(S), OpSum.LocalOperator(S), opsum(lat, b[1]))
        msg = argerr(() -> opsum(lat, b[1], bad))
        @test msg !== nothing
        @test argerr(() -> Terms(b[1], bad)) == msg
        @test argerr(() -> Terms(bad)) == msg
        @test argerr(() -> Terms{SU2Irrep}(bad)) == msg
        @test argerr(() -> Terms(b[1], [b[2], [bad]])) == msg
    end
    # an ExpSum has channels, which only an OperatorSum holds
    e = expterm(b[1]; decay = 0.3)
    for f in (() -> Terms(e), () -> Terms(b[1], e), () -> Terms(b[1], [b[2], e]), () -> Terms{SU2Irrep}(e))
        @test occursin("opsum", argerr(f))
    end
    @test opsum(lat, b[1], [b[2], e]).channels == e
    # one symmetry per collection
    other = OpSum.ExpSum{U1Irrep}()
    @test_throws ArgumentError opsum(lat, b[1], other)
    @test_throws ArgumentError Terms(b[1], other)
end
