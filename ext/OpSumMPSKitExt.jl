module OpSumMPSKitExt

# `instantiate(W::SymbolicMPO)`: MPSKit's Hamiltonians from OpSum's Jordan form. OpSum emits one
# `SparseBlockTensorMap` per site (`jordan_mpo_tensors`); MPSKit wants `JordanMPOTensor`s.

using OpSum: OpSum, SymbolicMPO, TaylorMPO, OperatorSum, FiniteChain, InfiniteChain, jordan_mpo_tensors,
    irrep_mpo_tensors, taylor_mpo, instantiate
using MPSKit: MPSKit, FiniteMPOHamiltonian, InfiniteMPOHamiltonian, FiniteMPO, InfiniteMPO, TaylorCluster
using BlockTensorKit: nonzero_pairs
using TensorKit: spacetype, storagetype, space

# `JordanMPOTensor(::SparseBlockTensorMap)` asserts that both diagonal corners are identities, which a
# finite chain's boundary tensors cannot satisfy. So every site goes through the `undef` + `setindex!` route
# (which also recognises `BraidingTensor`s), uniformly, as a mixture would give an abstractly typed vector.
function _jordan(W)
    O = MPSKit.jordanmpotensortype(spacetype(W), storagetype(W))(undef, space(W))
    for (I, v) in nonzero_pairs(W)
        O[I] = v
    end
    return O
end

OpSum._mpskit_mpo(W::SymbolicMPO{I, <:FiniteChain}) where {I} =
    FiniteMPOHamiltonian(map(_jordan, jordan_mpo_tensors(W)))
OpSum._mpskit_mpo(W::SymbolicMPO{I, <:InfiniteChain}) where {I} =
    InfiniteMPOHamiltonian(map(_jordan, jordan_mpo_tensors(W)))

# `instantiate(W::TaylorMPO, τ)`: the evaluated site tensors `B ⊗ V ← V ⊗ B` are MPSKit's `MPO` tensors as they
# stand (finite: one-dimensional boundary bonds).
OpSum._mpskit_mpo(W::TaylorMPO{I, <:FiniteChain}, τ::Number) where {I} = FiniteMPO(irrep_mpo_tensors(W, τ))
OpSum._mpskit_mpo(W::TaylorMPO{I, <:InfiniteChain}, τ::Number) where {I} = InfiniteMPO(irrep_mpo_tensors(W, τ))

# Route MPSKit's own entry point to the symbolic builder: `exp(-i dt H)`, or `exp(-dt H)` for imaginary time.
# `tol` is accepted for signature parity; the symbolic construction has nothing to drop.
function MPSKit.make_time_mpo(
        H::OperatorSum, dt::Number, alg::TaylorCluster;
        tol = nothing, imaginary_evolution::Bool = false
    )
    τ = imaginary_evolution ? -dt : -1im * dt
    return instantiate(taylor_mpo(H, alg.N; extension = alg.extension, compression = alg.compression), τ)
end

end
