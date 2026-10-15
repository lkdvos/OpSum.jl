using Test
using OpSum
using OpSum: instantiate, couple, opsum, mpo, mpo_tensormap
using TensorKit
using BlockTensorKit: BlockTensorKit, SparseBlockTensorMap, nonzero_keys, eachspace
using MPSKit
using MPSKit: JordanMPOTensor, FiniteMPOHamiltonian, FiniteMPS, expectation_value
using TensorKitTensors.SpinOperators: SpinOperators

include(joinpath(@__DIR__, "..", "testutils.jl"))                                   # LO

# Reference models, one per sector kind, at N = 4 (the 3-body one at 3): a dense contraction costs a
# fresh `ncon` specialization per (length, sectortype). Structural checks run on all, the dense oracle
# on `oracle_models`.
function reference_models()
    BS = spin_basis(SU2Irrep)
    SU2 = spin_ops(SU2Irrep)
    S = spinvector(BS)
    BU = spin_basis(U1Irrep)
    U1 = spin_ops(U1Irrep)
    BT = spin_basis()
    Triv = spin_ops()

    models = [
        # Jordan padding has to reproduce the textbook bond dim 3
        "heisenberg N=4" => (opsum(SU2.S_exchange[i, i + 1] for i in 1:3), FiniteChain(BS, 4)),
        # two disconnected trivial-charge channels at the middle bond
        "singlet pairs N=4" =>
            (opsum(SU2.S_exchange[1, 2], SU2.S_exchange[3, 4]), FiniteChain(BS, 4)),
        # the caterpillar inner line rides the bond
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
        # on-site terms: the only source of a D block
        "trivial chain + field N=4" => (
            opsum(
                (Triv.S_plus_S_min[i, i + 1] for i in 1:3),
                0.5 * Triv.S_z[2],
            ),
            FiniteChain(BT, 4),
        ),
    ]
    return [name => (opsum(sites, H), sites) for (name, (H, sites)) in models]
end

# The next-nearest-neighbour hop makes an odd-parity bond charge pass over a site, so the diagonal
# pass-through there is a braiding whose `BraidingTensor` has to carry the anticommutation sign.
function fermionic_model(N = 4)
    F = fermion_ops()
    cdc = F.f_plus_f_min                                # c†ᵢ cⱼ, in any site order
    H = opsum(
        (
            cdc[i, i + 1] + cdc[i + 1, i] + 2 * F.f_num[i] * F.f_num[i + 1]
                for i in 1:(N - 1)
        ),
        (0.3 * (cdc[i, i + 2] + cdc[i + 2, i]) for i in 1:(N - 2)),
    )
    return "t-V fermions N=$N" => (H, FiniteChain(fermion_basis(), N))
end

# The dense oracle runs where a sign or fusion coefficient could be wrong: SU(2) with a caterpillar
# inner line, and fermionic braiding. Both at N = 3, the smallest chain that has each feature.
function oracle_models()
    name, (H, sites) = fermionic_model(3)
    return [reference_models()[3], name => (opsum(sites, H), sites)]
end

# The Jordan contract: trivial boundary bonds, matching virtual spaces, identity corners, nothing
# entering the start channel from below or leaving the finish channel sideways.
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
            # the flag a consumer reads to keep identity blocks out of dense storage
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
    check_jordan_structure(Ws)
end

@testset "Jordan padding costs at most the two identity channels" begin
    for (_, (H, sites)) in reference_models()
        N = length(sites)
        _, secs = reduced(mpo(H))
        Ws = jordan_mpo_tensors(H)
        for i in 1:(N - 1)
            @test length(secs[i]) <= size(Ws[i], 4) <= length(secs[i]) + 2
        end
        @test size(Ws[N], 4) == 1
    end

    # textbook nearest-neighbour chain: padding is +1 at the first and last internal bonds, 0 in the bulk
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
    @test_throws ArgumentError jordan_mpo_tensors(opsum(FiniteChain(B, 3), S[i] for i in 1:3))
    @test_throws ArgumentError jordan_mpo_tensors(OperatorSum(FiniteChain(B, 3)))
    cell = jordan_mpo_tensors(opsum(InfiniteChain(B), b[1]))
    @test length(cell) == 1 && cell[1][1, 1, 1, 1] isa BraidingTensor
    H = opsum(FiniteChain(B, 4), b[i] for i in 1:3)
    alg = SVDBondAlgorithm(truncrank(1); sweep = SequentialSVD)
    @test_throws ArgumentError jordan_mpo_tensors(H, alg)
end

# The MPSKit seam: `JordanMPOTensor(::SparseBlockTensorMap)` asserts identity corners, which only bulk
# tensors satisfy, so it is exercised on the bulk and `instantiate(mpo(H))` handles the boundaries.
@testset "MPSKit round trip — $name" for (name, (H, sites)) in reference_models()
    Ws = jordan_mpo_tensors(H)

    for i in 2:(length(Ws) - 1)
        J = JordanMPOTensor(Ws[i])
        @test J isa MPSKit.JordanMPOTensor
        @test Set(nonzero_keys(J)) == Set(nonzero_keys(Ws[i]))
    end

    Hmpo = instantiate(mpo(H))
    @test Hmpo isa FiniteMPOHamiltonian
    @test length(Hmpo) == length(sites)
    # the A/B/C/D accessors silently drop an entry outside the Jordan pattern, so a misplacement shows here
    for (W, J) in zip(Ws, parent(Hmpo))
        @test SparseBlockTensorMap(J) ≈ W
    end
end

@testset "same operator as the dense oracle — $name" for (name, (H, sites)) in oracle_models()
    Ws = jordan_mpo_tensors(H)
    @test mpo_tensormap(map(TensorMap, Ws)) ≈ instantiate(H)
end

# Energies of the instantiated Hamiltonian against MPSKit's own construction from the local term, on a
# random state (a DMRG run would only add MPSKit-algorithm compilation to the same check).
@testset "MPSKit expectation value matches the local-operator construction" begin
    B = spin_basis(SU2Irrep)
    V = space(B)
    b = spin_ops(SU2Irrep).S_exchange
    N = 4
    H = opsum(FiniteChain(B, N), b[i] for i in 1:(N - 1))
    Hmpo = instantiate(mpo(H))
    ref = FiniteMPOHamiltonian(
        fill(V, N), ((i, i + 1) => SpinOperators.S_exchange(ComplexF64, SU2Irrep) for i in 1:(N - 1))...
    )
    # half-integer and integer spins: odd bonds of a spin-½ chain carry the former
    ψ = FiniteMPS(randn, ComplexF64, N, V, SU2Space(0 => 4, 1 // 2 => 4, 1 => 4))
    @test expectation_value(ψ, Hmpo) ≈ expectation_value(ψ, ref)
    @test abs(expectation_value(ψ, ref)) > 1.0e-3
end
