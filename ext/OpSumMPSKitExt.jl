module OpSumMPSKitExt

# `instantiate(W::SymbolicMPO)`: MPSKit's Hamiltonians from OpSum's Jordan form. OpSum emits one
# `SparseBlockTensorMap` per site (`jordan_mpo_tensors`); MPSKit wants `JordanMPOTensor`s.

using OpSum: OpSum, SymbolicMPO, FiniteChain, InfiniteChain, jordan_mpo_tensors
using MPSKit: MPSKit, FiniteMPOHamiltonian, InfiniteMPOHamiltonian
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

OpSum._mpskit_hamiltonian(W::SymbolicMPO{I, <:FiniteChain}) where {I} =
    FiniteMPOHamiltonian(map(_jordan, jordan_mpo_tensors(W)))
OpSum._mpskit_hamiltonian(W::SymbolicMPO{I, <:InfiniteChain}) where {I} =
    InfiniteMPOHamiltonian(map(_jordan, jordan_mpo_tensors(W)))

end
