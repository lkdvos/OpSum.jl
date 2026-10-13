# The Taylor construction of Van Damme, Haegeman, McCulloch, Vanderstraeten, "Efficient higher-order matrix
# product operators for time evolution" (SciPost Phys. 17, 135 (2024)): start from the `N`-fold Kronecker power
# of the Jordan-form MPO (`_mpo_power`, which represents `H^N`), fold the "finished" levels back onto the start
# level with the weights of the Taylor series (Alg. 1) and merge the levels that differ by a permutation (Alg. 2).
# The result is an MPO of `exp(τH)` correct through `τ^N`, whose entries are polynomials in `τ`.
#
# Conventions
#
#   * Level. A bond index of the power is a tuple of `N` Jordan indices, `1` the start channel and `n` (the
#     last index of the bond) the finish channel, the paper's `1` and `3`. A level is a position in
#     `levels[j]`, the levels of bond `j`, which is right of site `j` (on an infinite chain bond `L` is the one
#     left of site 1; the left boundary of a finite chain is the single level `(1, …, 1)`).
#   * Right boundary. A finite chain is powered *closed* (`_mpo_power(…; closed = true)`): its last bond has
#     the start channel too, so it has all `2^N` levels of `{1, n}^N` and Alg. 1 folds them like it does on any
#     bond. The Hamiltonian MPO itself lacks that channel only because its boundary vector selects the finish;
#     the Taylor MPO selects the folded level `(1, …, 1)` instead.
#   * Orientation. Alg. 2 merges level `c` into its canonical level `s`. The left environment of a level (the
#     operator accumulated to its left) is unchanged by moving the start channels among the layers, because they
#     contribute identities, and the right environment by moving the finish channels. So for a left-merge
#     (`c` has at least as many starts as finishes, `s` has them in front) the column `c` is dropped and the row
#     added to row `s`; for a right-merge (`s` has the finishes at the back) the row is dropped and the column
#     added to column `s`.
#   * Gauge. Permuting the components of a level induces `R(b_k, b_m)` per swapped pair of component charges
#     (`-1` for two fermion-odd ones). Alg. 2 only ever moves a start (or finish) component past the others,
#     and the identity channels have charge `unit`, so the gauge is trivial and the rows and columns are added
#     as they stand. (The dense operator of the merged MPO is the check: a wrong sign changes it.)
#
# Entries are `TaylorEntry`s: polynomials in `τ` with `SiteOperator` coefficients, so the construction is
# independent of `τ` and the numeric value is only taken in `evaluate`.

using SparseArrays: SparseMatrixCSC
using TensorKit: Sector

"""
    TaylorEntry{I}

A polynomial in `τ` with [`SiteOperator`](@ref) coefficients: parallel vectors of strictly ascending `powers`
and the nonzero `ops` multiplying them. `+` adds, `_scaled(e, c, k)` multiplies by `c τ^k`, `evaluate(e, τ)`
gives the [`SiteOperator`](@ref) at a value of `τ`.
"""
struct TaylorEntry{I <: Sector}
    powers::Vector{Int}
    ops::Vector{SiteOperator{I}}
end
TaylorEntry{I}() where {I} = TaylorEntry{I}(Int[], SiteOperator{I}[])
function TaylorEntry(op::SiteOperator{I}, power::Integer = 0) where {I}
    op = prune(op)
    return isempty(op) ? TaylorEntry{I}() : TaylorEntry{I}([Int(power)], [op])
end

Base.zero(::Union{TaylorEntry{I}, Type{TaylorEntry{I}}}) where {I} = TaylorEntry{I}()
Base.iszero(e::TaylorEntry) = isempty(e.powers)      # an entry is always kept pruned
Base.length(e::TaylorEntry) = length(e.powers)
Base.:(==)(x::TaylorEntry{I}, y::TaylorEntry{I}) where {I} = x.powers == y.powers && x.ops == y.ops
Base.hash(e::TaylorEntry, h::UInt) = hash(e.ops, hash(e.powers, hash(:TaylorEntry, h)))
function Base.isapprox(x::TaylorEntry{I}, y::TaylorEntry{I}; kwargs...) where {I}
    return x.powers == y.powers && all(isapprox(a, b; kwargs...) for (a, b) in zip(x.ops, y.ops))
end

function Base.:+(x::TaylorEntry{I}, y::TaylorEntry{I}) where {I}
    powers = Int[]
    ops = SiteOperator{I}[]
    i, j = 1, 1
    while i <= length(x) || j <= length(y)
        px = i <= length(x) ? x.powers[i] : typemax(Int)
        py = j <= length(y) ? y.powers[j] : typemax(Int)
        p = min(px, py)
        op = if px == py
            x.ops[i] + y.ops[j]
        else
            px < py ? x.ops[i] : y.ops[j]
        end
        i += px == p
        j += py == p
        op = prune(op)
        isempty(op) && continue
        push!(powers, p)
        push!(ops, op)
    end
    return TaylorEntry{I}(powers, ops)
end

# `c τ^k e`
function _scaled(e::TaylorEntry{I}, c::Number, k::Integer = 0) where {I}
    iszero(c) && return TaylorEntry{I}()
    return TaylorEntry{I}(e.powers .+ Int(k), [prune(op * c) for op in e.ops])
end

Base.:*(c::Number, e::TaylorEntry) = _scaled(e, c)

"""
    evaluate(e::TaylorEntry, τ::Number) -> SiteOperator
    evaluate(W::TaylorMPO, τ::Number) -> Vector{SparseMatrixCSC{SiteOperator}}

The value at `τ`: of one polynomial entry, or of every entry of a [`TaylorMPO`](@ref) (its bond matrices, the
form [`irrep_mpo_tensors`](@ref) takes).
"""
function evaluate(e::TaylorEntry{I}, τ::Number) where {I}
    acc = zero(SiteOperator{I})
    for (p, op) in zip(e.powers, e.ops)
        acc += op * τ^p
    end
    return prune(acc)
end

function Base.show(io::IO, e::TaylorEntry{I}) where {I}
    iszero(e) && return print(io, "TaylorEntry{", I, "}(0)")
    print(io, "TaylorEntry{", I, "}(")
    for (k, (p, op)) in enumerate(zip(e.powers, e.ops))
        k > 1 && print(io, " + ")
        print(io, "τ^", p, "*(", op, ")")
    end
    return print(io, ")")
end

# ---------------------------------------------------------------------------------------------------------
# The working matrix of one site: a dict of keys by row, with the columns' row sets alongside, so that both
# the row operations and the column operations of the algorithms cost their own size.

struct WorkMatrix{I <: Sector}
    rows::Vector{Dict{Int, TaylorEntry{I}}}     # rows[r][c]
    cols::Vector{Set{Int}}                      # cols[c] = the rows r with an entry at (r, c)
end
function WorkMatrix{I}(nrows::Int, ncols::Int) where {I <: Sector}
    return WorkMatrix{I}([Dict{Int, TaylorEntry{I}}() for _ in 1:nrows], [Set{Int}() for _ in 1:ncols])
end

# `M[r, c] += e`, an entry that cancels is removed
function _add!(M::WorkMatrix, r::Int, c::Int, e::TaylorEntry)
    iszero(e) && return M
    new = haskey(M.rows[r], c) ? M.rows[r][c] + e : e
    if iszero(new)
        delete!(M.rows[r], c)
        delete!(M.cols[c], r)
    else
        M.rows[r][c] = new
        push!(M.cols[c], r)
    end
    return M
end

# `M[dst, :] += M[src, :]`
function _addrow!(M::WorkMatrix, dst::Int, src::Int)
    for (c, e) in collect(M.rows[src])
        _add!(M, dst, c, e)
    end
    return M
end

# `M[:, dst] += c τ^k M[:, src]`
function _addcol!(M::WorkMatrix, dst::Int, src::Int, c::Number = 1, k::Integer = 0)
    for r in collect(M.cols[src])
        _add!(M, r, dst, _scaled(M.rows[r][src], c, k))
    end
    return M
end

function _droprow!(M::WorkMatrix, r::Int)
    for c in keys(M.rows[r])
        delete!(M.cols[c], r)
    end
    empty!(M.rows[r])
    return M
end

function _dropcol!(M::WorkMatrix, c::Int)
    for r in M.cols[c]
        delete!(M.rows[r], c)
    end
    empty!(M.cols[c])
    return M
end

# ---------------------------------------------------------------------------------------------------------

# The state of the construction: one working matrix per site (rows the levels of the bond to its left, columns
# those of the bond to its right) and, per bond, its levels with the ones not yet removed.
struct TaylorWork{I <: Sector, L <: AbstractLattice, N}
    lattice::L
    Ms::Vector{WorkMatrix{I}}
    levels::Vector{Vector{NTuple{N, Int}}}      # bond j is right of site j
    index::Vector{Dict{NTuple{N, Int}, Int}}
    top::Vector{Int}                            # the finish index of bond j (its number of Jordan indices)
    secs::Vector{Vector{I}}
    alive::Vector{BitVector}
end

_isfinite(::TaylorWork{I, L}) where {I, L} = L <: FiniteChain
nsites(w::TaylorWork) = length(w.Ms)

# the site whose rows are the levels of bond `j` (`0` at the right end of a finite chain)
_nextsite(w::TaylorWork, j::Int) = j < nsites(w) ? j + 1 : (_isfinite(w) ? 0 : 1)
# the bond whose levels are the rows of site `i` (`0` at the left end of a finite chain)
_rowbond(w::TaylorWork, i::Int) = i > 1 ? i - 1 : (_isfinite(w) ? 0 : nsites(w))

function _taylor_work(W::SymbolicMPO{I, L}, N::Integer) where {I, L}
    P = _mpo_power(W, N; closed = true)
    levels = P.levels
    Ms = map(P.mpo.Ws) do M
        wm = WorkMatrix{I}(size(M)...)
        for (idx, op) in storedpairs(M)
            r, c = Tuple(idx)
            _add!(wm, r, c, TaylorEntry(op))
        end
        return wm
    end
    return TaylorWork{I, L, Int(N)}(
        W.lattice, Ms, levels, [Dict(t => k for (k, t) in enumerate(lv)) for lv in levels],
        [maximum(maximum, lv) for lv in levels], P.mpo.bondsectors, [trues(length(lv)) for lv in levels]
    )
end

function _remove!(w::TaylorWork, j::Int, k::Int)
    _dropcol!(w.Ms[j], k)
    nxt = _nextsite(w, j)
    nxt == 0 || _droprow!(w.Ms[nxt], k)
    w.alive[j][k] = false
    return w
end

# Alg. 1: every level `b` of `{1, n}^N` other than `(1, …, 1)`, with `a` finish channels, is folded into the
# start level, `O[:, (1…1)] += τ^a (N - a)! / N! · O[:, b]`, and removed. Returns the number of levels removed.
function _fold_passive!(w::TaylorWork{I, L, N}) where {I, L, N}
    removed = 0
    for j in 1:nsites(w)
        n = w.top[j]
        w.levels[j][1] == ntuple(_ -> 1, Val(N)) || _invariant("the start level is not first on bond $j")
        for (k, t) in enumerate(w.levels[j])
            a = count(==(n), t)
            (a > 0 && all(l -> l == 1 || l == n, t)) || continue
            _addcol!(w.Ms[j], 1, k, factorial(N - a) / factorial(N), a)
            _remove!(w, j, k)
            removed += 1
        end
    end
    return removed
end

# `t` with the entries equal to `v` moved to the front (`front`) or the back, the others kept in order
function _gather(t::NTuple{N, Int}, v::Int, front::Bool) where {N}
    same, other = filter(==(v), t), filter(!=(v), t)
    return NTuple{N, Int}(front ? (same..., other...) : (other..., same...))
end

# Alg. 2 (exact): a level with at least as many start channels as finish channels is merged into the level with
# the starts in front, one with more finishes into the level with the finishes at the back, the others' order
# kept (see the file header for which of the row and the column is added and which dropped). Returns the
# number of levels removed.
function _merge_equivalent!(w::TaylorWork)
    removed = 0
    for j in 1:nsites(w)
        n = w.top[j]
        M = w.Ms[j]
        nxt = _nextsite(w, j)
        for (k, t) in enumerate(w.levels[j])
            w.alive[j][k] || continue
            leftmerge = count(==(n), t) <= count(==(1), t)
            s = leftmerge ? _gather(t, 1, true) : _gather(t, n, false)
            s == t && continue
            nxt == 0 && _invariant("level $t on the right boundary has to be merged: fold the passive levels first")
            target = w.index[j][s]
            leftmerge ? _addrow!(w.Ms[nxt], target, k) : _addcol!(M, target, k)
            _remove!(w, j, k)
            removed += 1
        end
    end
    return removed
end

# The levels reachable from the left boundary (`forward`) or from which the right boundary is, over the
# entries present. On an infinite chain the boundary is the level `(1, …, 1)` at the end of the unit cell, and
# the sets are the fixed point of going around it.
function _reachable(w::TaylorWork, forward::Bool)
    L = nsites(w)
    reach = [falses(length(lv)) for lv in w.levels]
    seed = forward ? _rowbond(w, 1) : L           # the bond with a boundary level, if any
    seed == 0 || (reach[seed][1] = true)
    changed = true
    while changed
        changed = false
        for i in (forward ? (1:L) : (L:-1:1))
            M = w.Ms[i]
            src, dst = forward ? (_rowbond(w, i), i) : (i, _rowbond(w, i))
            dst == 0 && continue
            for (r, row) in enumerate(M.rows), c in keys(row)
                from, to = forward ? (r, c) : (c, r)
                (src == 0 ? from == 1 : reach[src][from]) || continue
                if !reach[dst][to]
                    reach[dst][to] = true
                    changed = true
                end
            end
        end
    end
    return reach
end

# Remove the levels that no path from the left boundary reaches, or from which no path reaches the right
# boundary. Returns the number of levels removed.
function _prune_orphans!(w::TaylorWork)
    fwd, bwd = _reachable(w, true), _reachable(w, false)
    removed = 0
    for j in 1:nsites(w), k in eachindex(w.alive[j])
        w.alive[j][k] || continue
        (fwd[j][k] && bwd[j][k]) && continue
        k == 1 && _invariant("the start level of bond $j has no path through the chain")
        _remove!(w, j, k)
        removed += 1
    end
    return removed
end

# compact the surviving levels and finalise the matrices
function _taylor_finish(w::TaylorWork{I}, order::Int) where {I}
    newidx = [cumsum(a) for a in w.alive]
    Ws = map(1:nsites(w)) do i
        rb = _rowbond(w, i)
        cb = i
        nr = rb == 0 ? 1 : count(w.alive[rb])
        dict = Dictionary{CartesianIndex{2}, TaylorEntry{I}}()
        for (r, row) in enumerate(w.Ms[i].rows), (c, e) in row
            (rb == 0 || w.alive[rb][r]) || _invariant("an entry survives in a removed row")
            w.alive[cb][c] || _invariant("an entry survives in a removed column")
            insert!(dict, CartesianIndex(rb == 0 ? 1 : newidx[rb][r], newidx[cb][c]), e)
        end
        return sparse_from_dict(dict, (nr, count(w.alive[cb])))
    end
    secs = [w.secs[j][w.alive[j]] for j in 1:nsites(w)]
    return TaylorMPO{I, typeof(w.lattice)}(w.lattice, Ws, secs, order)
end
