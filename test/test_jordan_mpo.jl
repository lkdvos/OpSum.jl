using Test
using OpSum
using OpSum: instantiate, couple, opsum, mpo
using TensorKit
using BlockTensorKit: BlockTensorKit, SparseBlockTensorMap, nonzero_keys, eachspace
using MPSKit
using MPSKit: JordanMPOTensor, FiniteMPOHamiltonian, FiniteMPS, DMRG, find_groundstate,
    expectation_value
using LinearAlgebra: dot

include(joinpath(@__DIR__, "testutils.jl"))                                   # LO
include(joinpath(@__DIR__, "..", "benchmark", "ShowcaseModels.jl"))           # mpo_tensormap
using .ShowcaseModels: mpo_tensormap, spectrum

# Reference models, one per sector kind, all at N = 4 (except the 3-body one, which needs 3 sites).
# The size matters: contracting the assembled MPO against the `instantiate` oracle costs a fresh
# `ncon` specialization per (length, sectortype) pair, so every extra combination is a minute of
# compilation for no extra coverage. Structural checks run on all of these; the dense oracle runs on
# the three in `oracle_models`.
function reference_models()
    BS = spin_basis(SU2Irrep)
    SU2 = spin_ops(SU2Irrep)
    S = spinvector(BS)                                  # the charged spin vector, for the 3-body term
    BU = spin_basis(U1Irrep)
    U1 = spin_ops(U1Irrep)
    BT = spin_basis()
    Triv = spin_ops()

    models = [
        # SU(2) Heisenberg: the case where Jordan padding has to reproduce the textbook bond dim 3
        "heisenberg N=4" => (opsum(SU2.S_exchange[i, i + 1] for i in 1:3), FiniteChain(BS, 4)),
        # decoupled pairs: two disconnected trivial-charge channels at the middle bond
        "singlet pairs N=4" =>
            (opsum(SU2.S_exchange[1, 2], SU2.S_exchange[3, 4]), FiniteChain(BS, 4)),
        # a genuine 3-body term, so the caterpillar inner line rides the bond
        "K=3 SU(2) N=3" => (
            opsum(couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0))),
            FiniteChain(BS, 3),
        ),
        "U(1) hopping N=4" => (
            opsum(
                U1.S_plus_S_min[i, i + 1] + U1.S_min_S_plus[i, i + 1] for i in 1:3
            ),
            FiniteChain(BU, 4),
        ),
        fermionic_model(),
        # on-site terms: the only source of a D block, and of a finish channel at an early bond
        "trivial chain + field N=4" => (
            opsum(
                (Triv.S_plus_S_min[i, i + 1] for i in 1:3),
                0.5 * Triv.S_z[2],
            ),
            FiniteChain(BT, 4),
        ),
    ]
    # each model's terms go onto its chain: the `OperatorSum` is what the compression takes
    return [name => (opsum(sites, H), sites) for (name, (H, sites)) in models]
end

# t-V chain *with a next-nearest-neighbour hop*: that is what makes an odd-parity bond charge pass
# over a site, so the diagonal pass-through entry there is a braiding rather than a plain identity,
# and emitting it as a `BraidingTensor` has to carry the anticommutation sign.
function fermionic_model()
    F = fermion_ops()
    cdc = F.f_plus_f_min                                # c†ᵢ cⱼ, in any site order
    H = opsum(
        (
            cdc[i, i + 1] + cdc[i + 1, i] + 2 * F.f_num[i] * F.f_num[i + 1]
                for i in 1:3
        ),
        (0.3 * (cdc[i, i + 2] + cdc[i + 2, i]) for i in 1:2),
    )
    return "t-V fermions N=4" => (H, FiniteChain(fermion_basis(), 4))
end

# The two the *dense* oracle runs on: the SU(2) coupling and the fermionic braiding. Contracting an
# N-site MPO against `instantiate` is the one genuinely expensive check in this file — a fresh `ncon`
# specialization per (length, sectortype) — and everything else here is structural or per-site, so it
# is spent only where a sign or a fusion coefficient could be wrong. The remaining sectors are covered
# densely by `test_irrep_mpo_tensors.jl`, which validates the same letter/coupler assembly.
oracle_models() = reference_models()[[1, 5]]

# The Jordan contract, checked without reference to any consumer: one-dimensional trivial boundary
# bonds, matching virtual spaces across every bond, identity corners, nothing entering the start
# channel from below and nothing leaving the finish channel sideways.
function check_jordan_structure(Ws)
    N = length(Ws)
    @test size(Ws[1], 1) == 1
    @test size(Ws[N], 4) == 1
    for i in 1:(N - 1)
        @test space(Ws[i], 4)' == space(Ws[i + 1], 1)
    end
    for W in Ws
        rows, cols = size(W, 1), size(W, 4)
        keys4 = collect(nonzero_keys(W))
        if cols > 1
            @test CartesianIndex(1, 1, 1, 1) in keys4
            # a `BraidingTensor` and not merely something equal to one: that is the flag a consumer
            # reads to keep the ubiquitous identity blocks out of dense storage
            @test W[1, 1, 1, 1] isa BraidingTensor
            @test !any(K -> K[4] == 1 && K[1] != 1, keys4)
        end
        if rows > 1
            @test CartesianIndex(rows, 1, 1, cols) in keys4
            @test W[rows, 1, 1, cols] isa BraidingTensor
            @test !any(K -> K[1] == rows && K[4] != cols, keys4)
        end
    end
    return nothing
end

@testset "Jordan structure — $name" for (name, (H, sites)) in reference_models()
    Ws = jordan_mpo_tensors(H)
    @test Ws isa Vector{<:SparseBlockTensorMap}
    check_jordan_structure(Ws)
end

@testset "Jordan padding costs at most the two identity channels" begin
    # `mpo`'s cover is minimal among all MPOs with its sparsity pattern; the Jordan form is minimal
    # among *Jordan-form* MPOs, and the gap is exactly the padded start/finish channels.
    for (_, (H, sites)) in reference_models()
        N = length(sites)
        _, secs = reduced(mpo(H))
        Ws = jordan_mpo_tensors(H)
        for i in 1:(N - 1)
            @test length(secs[i]) <= size(Ws[i], 4) <= length(secs[i]) + 2
        end
        @test size(Ws[N], 4) == 1
    end

    # The textbook nearest-neighbour Hamiltonian is uniformly three-dimensional in Jordan form: the
    # padding is +1 at the first internal bond, +1 at the last, and 0 in the bulk.
    B = spin_basis(SU2Irrep)
    b = spin_ops(SU2Irrep).S_exchange
    N = 6
    sites = FiniteChain(B, N)
    H = opsum(sites, b[i] for i in 1:(N - 1))
    @test map(length, mpo(H).bondsectors) == [2, 3, 3, 3, 2, 1]
    Ws = jordan_mpo_tensors(H)
    @test [size(W, 4) for W in Ws] == [3, 3, 3, 3, 3, 1]
    @test [size(W, 1) for W in Ws] == [1, 3, 3, 3, 3, 3]
end

@testset "rejected inputs" begin
    B = spin_basis(SU2Irrep)
    b = spin_ops(SU2Irrep).S_exchange
    S = spinvector(B)
    # a non-scalar total charge has no identity at the right boundary
    @test_throws ArgumentError jordan_mpo_tensors(opsum(FiniteChain(B, 3), S[i] for i in 1:3))
    # an empty operator has no bond structure at all
    @test_throws ArgumentError jordan_mpo_tensors(OperatorSum(FiniteChain(B, 3)))
    # an infinite chain's unit cell has a Jordan form, with both identity channels live
    cell = jordan_mpo_tensors(opsum(InfiniteChain(B), b[1]))
    @test length(cell) == 1 && cell[1][1, 1, 1, 1] isa BraidingTensor
    # a truncation aggressive enough to empty a bond cannot carry the identity corners
    H = opsum(FiniteChain(B, 4), b[i] for i in 1:3)
    alg = SVDBondAlgorithm(truncrank(1); sweep = SequentialSVD)
    @test_throws ArgumentError jordan_mpo_tensors(H, alg)
end

# ── The MPSKit seam ───────────────────────────────────────────────────────────
# `JordanMPOTensor(::SparseBlockTensorMap)` is the constructor this emission targets, and it takes the
# emitted tensors as they are — but only in the *bulk*: it asserts that both diagonal corners are
# identities, which a boundary tensor cannot satisfy. So `instantiate(mpo(H))` (the MPSKit extension)
# assembles the Hamiltonian through the routing that constructor itself uses, and the constructor is
# exercised separately on the bulk.
@testset "MPSKit round trip — $name" for (name, (H, sites)) in reference_models()
    Ws = jordan_mpo_tensors(H)

    # `JordanMPOTensor(::SparseBlockTensorMap)` consumes the bulk tensors unmodified
    for i in 2:(length(Ws) - 1)
        J = JordanMPOTensor(Ws[i])
        @test J isa MPSKit.JordanMPOTensor
        @test Set(nonzero_keys(J)) == Set(nonzero_keys(Ws[i]))
    end

    Hmpo = instantiate(mpo(H))
    @test Hmpo isa FiniteMPOHamiltonian
    @test length(Hmpo) == length(sites)
    # Site by site, the round trip through MPSKit's block split is lossless. This is the sharp test of
    # the ordering, and cheaper than contracting the chain: the A/B/C/D accessors *silently drop* an
    # entry that sits outside the Jordan pattern, so a misplacement shows up here as a missing block.
    for (W, J) in zip(Ws, parent(Hmpo))
        @test SparseBlockTensorMap(J) ≈ W
    end
end

@testset "same operator as the dense oracle — $name" for (name, (H, sites)) in oracle_models()
    Ws = jordan_mpo_tensors(H)
    @test mpo_tensormap(map(TensorMap, Ws)) ≈ instantiate(H)
end

@testset "MPSKit DMRG ground state matches exact diagonalization" begin
    B = spin_basis(SU2Irrep)
    V = space(B)
    b = spin_ops(SU2Irrep).S_exchange
    N = 6
    H = opsum(FiniteChain(B, N), b[i] for i in 1:(N - 1))
    Hmpo = instantiate(mpo(H))

    exact = spectrum(H)[1]
    # half-integer *and* integer spins: an odd bond of a spin-½ chain carries the former
    ψ = FiniteMPS(randn, ComplexF64, N, V, SU2Space(0 => 8, 1 // 2 => 8, 1 => 8, 3 // 2 => 4))
    ψ, = find_groundstate(ψ, Hmpo, DMRG(; maxiter = 50, verbosity = 0))
    @test real(expectation_value(ψ, Hmpo)) ≈ exact rtol = 1.0e-6
end
