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
    taylor_mpo(H::OperatorSum, N::Integer; extension = true, compression = false) -> TaylorMPO
    taylor_mpo(W::SymbolicMPO, N::Integer; extension = true, compression = false) -> TaylorMPO

An MPO of `exp(τH)` correct through `τ^N` (`N ≥ 1`), after Van Damme, Haegeman, McCulloch and Vanderstraeten,
"Efficient higher-order matrix product operators for time evolution", SciPost Phys. 17, 135 (2024). The
convention is `U ≈ exp(τH)`: imaginary time `exp(-βH)` is `τ = -β`, real time `exp(-i t H)` is `τ = -i t`.
The result is a [`TaylorMPO`](@ref) with `τ` symbolic; [`evaluate`](@ref) it (or call it) at a value.

It is the `N`-fold Kronecker power of the Jordan form of `mpo(H)` (which represents `H^N`) with every level
`b ∈ {start, finish}^N` folded into the all-start level with the weight `τ^a (N - a)! / N!` of the `a`
finished layers, which restores the Taylor series (Alg. 1), and the levels that differ only by a permutation of
the start or finish channels merged, which is exact (Alg. 2). The all-start level is the one level of the
boundaries of a finite chain. The operator is size-extensive: it contains all disconnected products of terms of
`H` too. With `χ` the number of interior Jordan indices of the MPO of `H`, the bond dimension is `1 + χ` for
`N = 1`, `1 + 2χ + χ²` for `N = 2` (at most: a nilpotent letter can leave levels unreachable, which are removed).

Two keywords add terms of order `τ^{N+1}` to the same construction:

  - `extension` (Alg. 3): adds, to the entries between existing levels, the terms of `H^{N+1}` that only overlap
    `N` times, with their exact Taylor weights. The bond dimension is that of Alg. 1 + 2, and only the entries
    of `H^{N+1}` that are used are computed (never the whole power).
  - `compression` (Alg. 4): merges every level that has finish channels and no start channel into the level
    with those finishes turned into starts, with the weight `τ^n (N - n)! / N!` of its `n` finishes. This is
    *lossy* at order `τ^{N+1}` (the error stays `O(τ^{N+1})`), but combined with the extension the paper found it
    more precise than the plain construction, at a smaller bond dimension: `1 + χ`, `1 + χ + χ²`,
    `1 + 3χ + χ² + χ³`, `1 + 5χ + 4χ² + χ³ + χ⁴` for `N = 1, …, 4`.

On an [`InfiniteChain`](@ref) `H` is a generating set as in [`mpo`](@ref); the MPO is that of the unit cell.
Needs the total charge of `H` to be `unit`, an abelian symmetry, and, on a finite chain, no exponentially
decaying channels, like [`jordan_mpo_tensors`](@ref).
"""
function taylor_mpo(W::SymbolicMPO, N::Integer; extension::Bool = true, compression::Bool = false)
    passes = Function[]
    extension && push!(passes, _extend!)
    push!(passes, _fold_passive!, _merge_equivalent!)
    compression && push!(passes, _compress!)
    return _taylor_mpo(W, N, passes)
end
function taylor_mpo(H::OperatorSum, N::Integer; kwargs...)
    return taylor_mpo(mpo(H), N; kwargs...)
end

# The passes are `work -> count` and run in order; orphans are removed last.
function _taylor_mpo(W::SymbolicMPO, N::Integer, passes)
    work = _taylor_work(W, N)
    foreach(pass -> pass(work), passes)
    _prune_orphans!(work)
    return _taylor_finish(work, Int(N))
end
