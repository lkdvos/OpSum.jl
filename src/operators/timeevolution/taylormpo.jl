using SparseArrays: SparseMatrixCSC, nonzeros
using TensorKit: Sector, dim

"""
    TaylorMPO{I<:Sector, L<:AbstractLattice}

An MPO of the time-evolution operator `exp(τH)`, correct through `τ^order`, with the time left symbolic: the
bond matrices `Ws[i]` of site `i` hold polynomials in `τ` (`TaylorEntry`s), and `bondsectors[i]` is the charge
of each index of the bond right of site `i`, as in [`SymbolicMPO`](@ref) (on an [`InfiniteChain`](@ref) bond
`L` is the one left of site 1; the right boundary of a finite chain is the single index of charge `unit`).
Built by [`taylor_mpo`](@ref); [`evaluate`](@ref)`(W, τ)`, or `W(τ)`, gives the bond matrices at a value of `τ`
and `irrep_mpo_tensors(W, τ)` the symmetric site tensors.
"""
struct TaylorMPO{I <: Sector, L <: AbstractLattice}
    lattice::L
    Ws::Vector{SparseMatrixCSC{TaylorEntry{I}, Int}}
    bondsectors::Vector{Vector{I}}
    order::Int
end

Base.length(W::TaylorMPO) = length(W.Ws)
Base.getindex(W::TaylorMPO, i::Integer) = W.Ws[i]

function Base.show(io::IO, W::TaylorMPO{I, L}) where {I, L}
    D = [sum(dim, sec; init = 0) for sec in W.bondsectors]
    return print(io, "TaylorMPO{", I, "}(", nameof(L), ", order = ", W.order, ", N = ", length(W), ", D = ", D, ")")
end

function evaluate(W::TaylorMPO{I}, τ::Number) where {I}
    return map(W.Ws) do M
        vals = SiteOperator{I}[evaluate(e, τ) for e in nonzeros(M)]
        return SparseMatrixCSC(size(M, 1), size(M, 2), copy(M.colptr), copy(M.rowval), vals)
    end
end

(W::TaylorMPO)(τ::Number) = evaluate(W, τ)

# the evaluated bond matrices carry no identity channels (`start`/`finish` empty: unknown), only the tiling
irrep_mpo_tensors(W::TaylorMPO, τ::Number) =
    irrep_mpo_tensors(SymbolicMPO(W.lattice, evaluate(W, τ), W.bondsectors, Int[], Int[]))

"""
    taylor_mpo(H::OperatorSum, N::Integer) -> TaylorMPO
    taylor_mpo(W::SymbolicMPO, N::Integer) -> TaylorMPO

An MPO of `exp(τH)` correct through `τ^N` (`N ≥ 1`), after Van Damme, Haegeman, McCulloch and Vanderstraeten,
"Efficient higher-order matrix product operators for time evolution", SciPost Phys. 17, 135 (2024). The
convention is `U ≈ exp(τH)`: imaginary time `exp(-βH)` is `τ = -β`, real time `exp(-i t H)` is `τ = -i t`.
The result is a [`TaylorMPO`](@ref) with `τ` symbolic; [`evaluate`](@ref) it (or call it) at a value.

It is the `N`-fold Kronecker power of the Jordan form of `mpo(H)` (which represents `H^N`) with every level
`b ∈ {start, finish}^N` folded into the all-start level with the weight `τ^a (N - a)! / N!` of the `a`
finished layers, which restores the Taylor series, and the levels that differ only by a permutation of the
start or finish channels merged, which is exact. The all-start level is the one level of the boundaries of a
finite chain. The operator is size-extensive: it contains all disconnected products of terms of `H` too.

On an [`InfiniteChain`](@ref) `H` is a generating set as in [`mpo`](@ref); the MPO is that of the unit cell.
Needs the total charge of `H` to be `unit`, an abelian symmetry, and, on a finite chain, no exponentially
decaying channels, like [`jordan_mpo_tensors`](@ref).
"""
taylor_mpo(W::SymbolicMPO, N::Integer) = _taylor_mpo(W, N, (_fold_passive!, _merge_equivalent!))
taylor_mpo(H::OperatorSum, N::Integer) = taylor_mpo(mpo(H), N)

# The passes are `work -> removed levels` and run in order; orphans are removed last.
function _taylor_mpo(W::SymbolicMPO, N::Integer, passes)
    work = _taylor_work(W, N)
    foreach(pass -> pass(work), passes)
    _prune_orphans!(work)
    return _taylor_finish(work, Int(N))
end
