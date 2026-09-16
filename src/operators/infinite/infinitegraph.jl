# Reduced MPO for an infinite chain with a repeating unit cell: a generating `Terms` bag over `L` sites stands for
# `Σ_{n∈ℤ} translate(H, n·L)`. Per bond this is the finite sweep's problem with one difference: the
# bond basis has to close on itself (`V_0 ≅ V_L`), so both identity channels are live at every bond.
# See `research/infinite-mpo.md` for the design.
#
# `_infinite_window` unrolls `2P+1` cells (`P·L ≥ R`), runs the unchanged finite sweep, and reads the
# middle cell off — exact in the bulk since every crossing (or collision-prone pending) translate lies
# entirely inside the window. Closure is checked, not assumed: bond `offset` and bond `offset+L` must
# agree as ordered bases and the extracted cell must equal the next one entry for entry.
#
# The two identity channels are found on the assembled cell by direction: nothing enters the start
# channel but itself, nothing leaves the done channel but itself.

using SparseArrays: SparseMatrixCSC
using TensorKit: Sector, unit, block, oneunit, ncon, removeunit, permute, @tensor
using .IrrepTensorOperators: IrrepOperator

# --- the extracted unit cell ------------------------------------------------------------------

"""
    InfiniteMPO{I<:Sector}

A reduced MPO for an infinite chain with a unit cell of `L = length(Ws)` sites.

`Ws[j]` is the reduced bond matrix of site `j` (`Ws[j] : bond j-1 → bond j`, entries are ITO letters
times reduced coefficients) and `bondsectors[j]` gives the bond charge of each index of bond `j`, the
bond to the *right* of site `j`. Bond `0` is bond `L`: `size(Ws[j], 1) == length(bondsectors[mod1(j-1, L)])`.

`start[j]` and `done[j]` are the indices of the two identity channels on bond `j` — the left boundary
vector is `e_{start[L]}` and the right boundary vector is `e_{done[L]}`.

The `(Ws, bondsectors)` pair has exactly the same types as the finite [`irrep_mpo`](@ref) output, so
`irrep_mpo_tensors` consumes it unchanged.
"""
struct InfiniteMPO{I <: Sector}
    Ws::Vector{SparseMatrixCSC{SiteOperator{I}, Int}}
    bondsectors::Vector{Vector{I}}
    start::Vector{Int}
    done::Vector{Int}
end

Base.length(H::InfiniteMPO) = length(H.Ws)

# iterate/destructure as `(Ws, bondsectors)`, matching the finite contract
Base.iterate(H::InfiniteMPO, state::Int = 1) =
    state == 1 ? (H.Ws, 2) : state == 2 ? (H.bondsectors, 3) : nothing

function Base.show(io::IO, H::InfiniteMPO{I}) where {I}
    D = [sum(dim, sec; init = 0) for sec in H.bondsectors]
    return print(io, "InfiniteMPO{", I, "}(L = ", length(H), ", D = ", D, ")")
end

# --- identity-channel detection ---------------------------------------------------------------

# Rows/columns of `W` that hold exactly one stored entry, and that entry is the bare pass-through
# (`isone`, i.e. the pass-through sentinel letter with unit coefficient — the identity backbone):
# `rowsucc[r] = c` if row `r`'s only entry is a bare pass-through into column `c` (else 0), and
# `colpred[c] = r` dually. Bond dimensions are small, so a dense scan over stored entries is fine.
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

# Follow a per-bond successor map once around the cell and keep the indices that come back to
# themselves — the closed identity backbones.
function _closed_chains(step::Vector{Vector{Int}}, order::AbstractVector{Int}, nstart::Int)
    chains = Vector{Vector{Int}}()
    for d0 in 1:nstart
        chain = Vector{Int}(undef, length(order))
        d = d0
        ok = true
        for (k, j) in enumerate(order)
            d = step[j][d]
            if iszero(d)
                ok = false
                break
            end
            chain[k] = d
        end
        (ok && d == d0) && push!(chains, chain)
    end
    return chains
end

"""
    _identity_channels(Ws, bondsectors) -> (start, done)

Locate the two identity channels of a closed unit cell. `start[j]` / `done[j]` are bond indices on
bond `j` (`j == L` is also bond `0`).

Both channels are chains of bare pass-through entries running all the way around the cell, told apart
by direction: the start channel's *column* has no other entry (nothing enters it), the done channel's
*row* has no other entry (nothing leaves it). Throws if either is missing or ambiguous — for the
lossless bipartite compression an infinite MPO without a unique identity backbone on each side has no
well-defined pair of boundary vectors.
"""
function _identity_channels(
        Ws::Vector{<:SparseMatrixCSC}, bondsectors::Vector{Vector{I}}
    ) where {I}
    L = length(Ws)
    D = [length(sec) for sec in bondsectors]
    succ = Vector{Vector{Int}}(undef, L)   # bond j-1 index -> bond j index (row-unique backbone)
    pred = Vector{Vector{Int}}(undef, L)   # bond j index   -> bond j-1 index (col-unique backbone)
    for j in 1:L
        succ[j], pred[j] = _identity_links(Ws[j], D[mod1(j - 1, L)], D[j])
    end

    # done: forwards from bond L around to bond L again, keeping row-unique bare pass-throughs
    dones = _closed_chains(succ, 1:L, D[L])
    # start: backwards from bond L, keeping column-unique bare pass-throughs; the chain is collected
    # in reverse bond order, so rotate it back to `start[j]`
    starts = map(_closed_chains(pred, reverse(1:L), D[L])) do rev
        # `rev[k]` is the index on bond `L-k`; index on bond L is the seed, i.e. `rev[end]`
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

# Canonical form of a reduced entry: its `letter => coefficient` list sorted by letter. `SiteOperator`
# does have a structural `==`, but it compares the two parallel vectors *in order* and `+` accumulates
# in insertion order, so two separately-built copies of the same entry can disagree on letter order.
# The fixed-point test compares entries built at different points in the sweep, so the sort is needed.
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

"""
    _fixedpoint_cell(Ws, bondsectors, L, R) -> Int

Index of the first unit cell of an unrolled sweep that has reached the translation-invariant fixed
point, or `0` if none has.

A cell qualifies when it and the *two* cells after it are identical site tensors and its two bond
charge lists agree — i.e. the sweep has repeated itself twice over, which pins both the bond bases and
the residual coefficients riding on them. Two consecutive repetitions rather than one because a single
coincidence is cheap to rule out and the window has cells to spare; cells closer than `R` to the left
end are skipped outright, since the window drops the translates that stick out there and a match in
that region would not be the periodic problem.
"""
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

# The finite-range terms and the geometric channels of a generating set, whichever of the two shapes it
# comes in: the sweep takes the first through the term table and the second alongside it.
_finitepart(gen::Terms) = gen
_finitepart(gen::MixedSum) = gen.terms
_geometricpart(::Terms{I}) where {I} = ExpSum{I}()
_geometricpart(gen::MixedSum) = gen.channels

"""
    _infinite_window(gen::Terms, lat::InfiniteChain; ncells = nothing) -> InfiniteMPO
    _infinite_window(gen::MixedSum, lat::InfiniteChain; ncells = nothing) -> InfiniteMPO

Reduced MPO for the infinite chain generated by the canonical unit-cell term sum `gen`, via the
unrolled-window construction described at the top of this file.

The window is unrolled and then *searched* for the fixed point rather than sized by a formula: the
number of cells it takes for the sweep to become translation-invariant is bounded by the interaction
range `R` for the bond bases, but the residual coefficients riding on a bond take a further reset to
settle (the identity backbone on the right does not exist until something has finished), so the honest
thing is to look. If no fixed point is found the window doubles, twice, before giving up.

`ncells` overrides the initial window size; it is exposed so a test can check the answer does not
depend on it.
"""
function _infinite_window(
        gen::Union{Terms{I}, MixedSum{I}}, lat::InfiniteChain;
        ncells::Union{Nothing, Int} = nothing
    ) where {I}
    L = length(lat)
    R = maxspan(gen)
    nc = something(ncells, 2 * (cld(R, L) + 3) + 3)
    local Ws, bondsectors, c
    for attempt in 1:3
        attempt == 1 || (nc *= 2)
        N = nc * L
        tt = ITOTermTable(window_terms(_finitepart(gen), lat, nc), N)
        Ws, bondsectors = _irrep_sweep(
            tt, N, VertexCover(); channels = _lower_channels(_geometricpart(gen), L)
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
    return InfiniteMPO{I}(cellWs, cellsecs, start, done)
end

# --- faithfulness ---------------------------------------------------------------------------------

"""
    tile(H::InfiniteMPO, ncells::Int) -> (Ws, bondsectors)

Unroll `ncells` copies of the unit cell into the finite `(Ws, bondsectors)` contract, over
`ncells * L` sites. The result is a finite MPO only once boundary vectors are chosen: `H.start[L]` on
the left and `H.done[L]` on the right.
"""
function tile(H::InfiniteMPO{I}, ncells::Int) where {I}
    L = length(H)
    N = ncells * L
    return ([H.Ws[mod1(j, L)] for j in 1:N], [H.bondsectors[mod1(j, L)] for j in 1:N])
end

"""
    mpo_terms_window(H::InfiniteMPO, lat::InfiniteChain, ncells::Int) -> Terms

Reconstruct the terms an infinite MPO generates inside a window of `ncells` unit cells: tile the cell,
then enumerate the paths that enter on the start channel and leave on the done channel.

This is the faithfulness check for the infinite construction, and unlike the bond dimensions it pins
the coefficients *and* the caterpillar fusion trees. It is a **sandwich**, not an equality, against
[`window_terms`](@ref):

* every term produced is a translate with support inside the window, at its exact coefficient — no
  spurious terms and no wrong weights;
* every translate whose support ends at least `R + 1` sites before the right edge is produced.

The gap is real and is a property of the compression, not a defect. A term's reduced coefficient does
not have to sit on its own last site: when its suffix class is *shared* with a longer term the
coefficient is folded onto the shared channel's trailing pass-through instead. The minimal example is
the pending↔started collision (`research/persistent-graph-mpo.md` §2.2) — an on-site field alongside a
two-site interaction, where the field's letter goes down at site `s` and its coefficient is picked up
on the identity at site `s + 1`. So a *path* can run up to `R` sites past the support of the term it
represents, and paths that would run off the right edge of a finite window are simply absent from it.
"""
function mpo_terms_window(H::InfiniteMPO{I}, lat::InfiniteChain, ncells::Int) where {I}
    L = length(H)
    Ws, secs = tile(H, ncells)
    return mpo_terms(
        Ws, secs; leftidx = H.start[L], rightidx = H.done[L]
    )
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

Contract a chain of MPO site tensors down to the operator it represents between two chosen bond-basis
vectors of the shared boundary bond (charges `bondsec`) — for an infinite MPO, the start and done
channels.

This is the infinite counterpart of the `mpo_tensormap` helper in `examples/common.jl`: that one drops
a one-dimensional vacuum boundary with `removeunit`, whereas here both boundary bonds are the full,
many-dimensional periodic bond, so they are first capped with one-hot maps. The result is in
`instantiate`'s convention (`o₁…o_N ← i₁…i_N ⊗ trivial`), so it can be compared against a dense oracle
directly. Exponential in `N`; for tests only.
"""
function contract_open(Ts, bondsec::Vector{I}, leftidx::Int, rightidx::Int) where {I}
    N = length(Ts)
    capped = copy(collect(Ts))
    capped[1] = _cap_left(_channel_cap(bondsec, leftidx, :left), capped[1])
    capped[N] = _cap_right(_channel_cap(bondsec, rightidx, :right), capped[N])
    net = [[i == 1 ? -(2N + 1) : i - 1, -i, -(N + i), i == N ? -(2N + 2) : i] for i in 1:N]
    O = ncon(capped, net)                # legs: o₁…o_N, i₁…i_N, bL, bR
    O = removeunit(O, 2N + 1)            # the capped left boundary
    return permute(O, (ntuple(identity, N), (ntuple(i -> N + i, N)..., 2N + 1)))
end
