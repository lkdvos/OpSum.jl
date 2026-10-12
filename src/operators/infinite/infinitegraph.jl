# Reduced MPO for an infinite chain: a generating bag over `L` sites stands for `Σ_n translate(H, n·L)`.
# This is the finite sweep with one difference: the bond basis closes on itself (`V_0 ≅ V_L`), so both
# identity channels are live at every bond. Design: `research/infinite-mpo.md`.
#
# `_infinite_window` unrolls a window of cells, runs the unchanged finite sweep, and reads off a cell
# that has reached the translation-invariant fixed point. Closure is checked, not assumed.

using SparseArrays: SparseMatrixCSC
using TensorKit: Sector, unit, block, oneunit, ncon, removeunit, permute, @tensor
using .IrrepTensorOperators: IrrepOperator

# --- identity-channel detection ---------------------------------------------------------------

# `rowsucc[r] = c` if row `r`'s only entry is a bare pass-through (`isone`) into column `c` (else 0),
# and `colpred[c] = r` dually: the identity backbone.
function _identity_links(W::SparseMatrixCSC, nrows::Int, ncols::Int)
    rowsucc, rowdeg = zeros(Int, nrows), zeros(Int, nrows)
    colpred, coldeg = zeros(Int, ncols), zeros(Int, ncols)
    for (idx, op) in storedpairs(W)
        r, c = Tuple(idx)
        rowdeg[r] += 1
        coldeg[c] += 1
        if isone(op)
            rowsucc[r] = c
            colpred[c] = r
        end
    end
    for r in 1:nrows
        rowdeg[r] == 1 || (rowsucc[r] = 0)
    end
    for c in 1:ncols
        coldeg[c] == 1 || (colpred[c] = 0)
    end
    return rowsucc, colpred
end

# Follow a per-bond successor map once around the cell; keep the indices that return to themselves.
function _closed_chains(step::Vector{Vector{Int}}, order::AbstractVector{Int}, nstart::Int)
    chains = Vector{Vector{Int}}()
    for d0 in 1:nstart
        chain = Vector{Int}(undef, length(order))
        d = d0
        for (k, j) in enumerate(order)
            d = step[j][d]
            iszero(d) && break
            chain[k] = d
        end
        d == d0 && push!(chains, chain)
    end
    return chains
end

# Locate the two identity channels of a closed cell: `start[j]` / `done[j]` are bond indices on bond `j`
# (`j == L` is also bond `0`). Both are closed chains of bare pass-throughs, told apart by direction:
# nothing enters the start channel but itself, nothing leaves the done channel but itself. Throws if
# either is missing or ambiguous.
function _identity_channels(
        Ws::Vector{<:SparseMatrixCSC}, bondsectors::Vector{Vector{I}}
    ) where {I}
    L = length(Ws)
    D = [length(sec) for sec in bondsectors]
    links = [_identity_links(Ws[j], D[mod1(j - 1, L)], D[j]) for j in 1:L]
    succ = first.(links)   # bond j-1 index -> bond j index
    pred = last.(links)    # bond j index   -> bond j-1 index

    dones = _closed_chains(succ, 1:L, D[L])
    # backwards from bond L; `rev[k]` is the index on bond `L-k` and `rev[end]` the one on bond L
    starts = map(_closed_chains(pred, reverse(1:L), D[L])) do rev
        return [k == L ? rev[end] : rev[L - k] for k in 1:L]
    end

    length(starts) == 1 || throw(
        ArgumentError(
            "expected exactly one start channel (a closed pass-through backbone that nothing " *
                "enters), found $(length(starts)): $starts"
        )
    )
    length(dones) == 1 || throw(
        ArgumentError(
            "expected exactly one done channel (a closed pass-through backbone that nothing " *
                "leaves), found $(length(dones)): $dones"
        )
    )
    s, d = only(starts), only(dones)
    s == d && throw(ArgumentError("the start and done channels coincide ($s); the MPO is degenerate"))
    for j in 1:L
        bondsectors[j][s[j]] == unit(I) ||
            throw(ArgumentError("start channel on bond $j carries charge $(bondsectors[j][s[j]]), expected $(unit(I))"))
        bondsectors[j][d[j]] == unit(I) ||
            throw(ArgumentError("done channel on bond $j carries charge $(bondsectors[j][d[j]]), expected $(unit(I))"))
    end
    return s, d
end

# --- the window construction ------------------------------------------------------------------

# Reduced entry as a `letter => coefficient` list sorted by letter: `SiteOperator`'s `==` is
# order-sensitive, and entries built at different points of the sweep may differ in letter order.
_canonform(op::SiteOperator) = sort!(collect(pairs(op)); by = first)

# Entry-for-entry equality of two reduced bond matrices.
function _entriesequal(A::SparseMatrixCSC, B::SparseMatrixCSC)
    size(A) == size(B) || return false
    pa = Dict(idx => _canonform(op) for (idx, op) in storedpairs(A))
    pb = Dict(idx => _canonform(op) for (idx, op) in storedpairs(B))
    keys(pa) == keys(pb) || return false
    return all(pa[k] == pb[k] for k in keys(pa))
end

# Do cells `c` and `c+1` of an unrolled sweep carry identical site tensors?
function _cellsequal(Ws::Vector{<:SparseMatrixCSC}, L::Int, c::Int)
    a, b = (c - 1) * L, c * L
    return all(j -> _entriesequal(Ws[a + j], Ws[b + j]), 1:L)
end

# Index of the first unit cell that has reached the translation-invariant fixed point (0 if none): it
# and the two cells after it are identical site tensors and its two bond charge lists agree. Cells
# closer than `R` to the left end are skipped, since the window drops translates sticking out there.
function _fixedpoint_cell(
        Ws::Vector{<:SparseMatrixCSC}, bondsectors::Vector{Vector{I}}, L::Int, R::Int
    ) where {I}
    ncells = length(Ws) ÷ L
    for c in (cld(R, L) + 2):(ncells - 2)
        bondsectors[(c - 1) * L] == bondsectors[c * L] || continue
        (_cellsequal(Ws, L, c) && _cellsequal(Ws, L, c + 1)) || continue
        return c
    end
    return 0
end

# Reduced MPO of the infinite chain generated by the unit-cell sum `gen` (see `unitcell_terms`).
# The window is unrolled and *searched* for the fixed point rather than sized by a formula (the
# residual coefficients on a bond take a further reset to settle); if none is found it doubles, twice.
# `ncells` overrides the initial size, so a test can check the answer does not depend on it.
function _infinite_window(
        gen::OperatorSum{I, <:InfiniteChain};
        ncells::Union{Nothing, Int} = nothing
    ) where {I}
    lat = gen.lattice
    L = length(lat)
    R = maxspan(gen)
    nc = something(ncells, 2 * (cld(R, L) + 3) + 3)
    local Ws, bondsectors, c
    for attempt in 1:3
        attempt == 1 || (nc *= 2)
        N = nc * L
        tt = ITOTermTable(window_terms(gen.terms, lat, nc), N)
        Ws, bondsectors = _irrep_sweep(
            tt, N, VertexCover(); channels = _lower_channels(gen.channels, L)
        )
        c = _fixedpoint_cell(Ws, bondsectors, L, R)
        iszero(c) || break
    end
    iszero(c) && throw(
        ErrorException(
            "the unrolled sweep did not reach a translation-invariant fixed point within $nc unit " *
                "cells (L = $L, interaction range R = $R). Two bonds one unit cell apart never " *
                "produced identical reduced tensors, so the compression is not translation-covariant " *
                "for this model — please report it."
        )
    )

    offset = (c - 1) * L
    cellWs = Ws[(offset + 1):(offset + L)]
    cellsecs = bondsectors[(offset + 1):(offset + L)]
    start, done = _identity_channels(cellWs, cellsecs)
    return SymbolicMPO{I, typeof(lat)}(lat, cellWs, cellsecs, start, done)
end

# --- faithfulness ---------------------------------------------------------------------------------

"""
    tile(H::SymbolicMPO, ncells::Int) -> (Ws, bondsectors)

Unroll `ncells` copies of an infinite chain's unit cell into the finite `(Ws, bondsectors)` contract.
Finite only once boundary vectors are chosen: `H.start[L]` on the left, `H.finish[L]` on the right.
"""
function tile(H::SymbolicMPO{I, <:InfiniteChain}, ncells::Int) where {I}
    L = length(H)
    N = ncells * L
    return ([H.Ws[mod1(j, L)] for j in 1:N], [H.bondsectors[mod1(j, L)] for j in 1:N])
end

"""
    mpo_terms_window(H::SymbolicMPO, ncells::Int) -> Terms

The terms an infinite MPO generates inside a window of `ncells` cells: tile the cell and enumerate the
paths from the start to the finish channel. This pins coefficients *and* fusion trees, and is a
**sandwich** against [`window_terms`](@ref), not an equality:

* every term produced is a translate inside the window at its exact coefficient;
* every translate ending at least `R + 1` sites before the right edge is produced.

The gap is a property of the compression: when a term's suffix class is shared with a longer term, its
coefficient is folded onto the shared channel's trailing pass-through (the pending/started collision,
`research/persistent-graph-mpo.md` §2.2), so a path can run up to `R` sites past the term's support.
"""
function mpo_terms_window(H::SymbolicMPO{I, <:InfiniteChain}, ncells::Int) where {I}
    L = length(H)
    Ws, secs = tile(H, ncells)
    return mpo_terms(Ws, secs; leftidx = H.start[L], rightidx = H.finish[L])
end

_cap_left(cap, W) = @tensor out[a o; i b] := cap[a; x] * W[x o; i b]
_cap_right(cap, W) = @tensor out[a o; i b] := W[a o; i x] * cap[x; b]

# One-hot map onto index `idx` of a bond, as a `oneunit ← B` (left) or `B ← oneunit` (right) tensor.
function _channel_cap(bondsec::Vector{I}, idx::Int, side::Symbol) where {I}
    bondsec[idx] == unit(I) ||
        throw(ArgumentError("boundary index $idx carries charge $(bondsec[idx]), expected $(unit(I))"))
    B = _bond_space(bondsec)
    deg = _deg_indices(bondsec)[idx]
    if side === :left
        cap = zeros(ComplexF64, oneunit(B) ← B)
        block(cap, unit(I))[1, deg] = 1
    else
        cap = zeros(ComplexF64, B ← oneunit(B))
        block(cap, unit(I))[deg, 1] = 1
    end
    return cap
end

"""
    contract_open(Ts, bondsec, leftidx, rightidx) -> AbstractTensorMap

Contract site tensors down to the operator between two bond-basis vectors of the shared boundary bond
(for an infinite MPO, the start and done channels), by capping both ends with one-hot maps. The result
is in `instantiate`'s convention. Exponential in `N`; for tests only.
"""
function contract_open(Ts, bondsec::Vector{I}, leftidx::Int, rightidx::Int) where {I}
    N = length(Ts)
    capped = copy(collect(Ts))
    capped[1] = _cap_left(_channel_cap(bondsec, leftidx, :left), capped[1])
    capped[N] = _cap_right(_channel_cap(bondsec, rightidx, :right), capped[N])
    return mpo_tensormap(capped)
end
