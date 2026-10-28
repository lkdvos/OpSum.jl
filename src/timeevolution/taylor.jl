# The Taylor construction of Van Damme, Haegeman, McCulloch, Vanderstraeten, "Efficient higher-order matrix
# product operators for time evolution" (SciPost Phys. 17, 135 (2024)): start from the `N`-fold Kronecker power
# of the Jordan-form MPO (`mpo_power`, which represents `H^N`), fold the "finished" levels back onto the start
# level with the weights of the Taylor series (Alg. 1) and merge the levels that differ by a permutation (Alg. 2).
# The result is an MPO of `exp(τH)` correct through `τ^N`, whose entries are polynomials in `τ`. Two optional
# passes go beyond that: the extension (Alg. 3) adds the order `N + 1` terms that reuse the existing levels, and
# the approximate compression (Alg. 4) merges the levels that agree through `τ^N`.
#
# Conventions
#
#   * Level. A bond index of the power is a tuple of `N` Jordan indices, `1` the start channel and `n` (the
#     last index of the bond) the finish channel, the paper's `1` and `3`. A level is a position in
#     `levels[j]`, the levels of bond `j`, which is right of site `j` (on an infinite chain bond `L` is the one
#     left of site 1; the left boundary of a finite chain is the single level `(1, …, 1)`).
#   * Right boundary. A finite chain is powered *closed* (`mpo_power(…; closed = true)`): its last bond has
#     the start channel too, so it has all `2^N` levels of `{1, n}^N` and Alg. 1 folds them like it does on any
#     bond. The Hamiltonian MPO itself lacks that channel only because its boundary vector selects the finish;
#     the Taylor MPO selects the folded level `(1, …, 1)` instead.
#   * Orientation. Alg. 2 merges level `c` into its canonical level `s`. The left environment of a level (the
#     operator accumulated to its left) is unchanged by moving the start channels among the layers, because they
#     contribute identities, and the right environment by moving the finish channels. So for a left-merge
#     (`c` has at least as many starts as finishes, `s` has them in front) the column `c` is dropped and the row
#     added to row `s`; for a right-merge (`s` has the finishes at the back) the row is dropped and the column
#     added to column `s`.
#   * Order of the passes. The paper applies the extension to the power `H^N` itself, before Alg. 1: it adds to
#     the entries `O[a, b]` with `1 ∉ b`, among them the all-finish column that Alg. 1 then folds (this is where
#     the `τ³ DDD` of the `(1,1)` entry of Eq. (43) comes from), and skips the rows that Alg. 1 removes. The
#     passes here run in the same order: extension, fold, merge, compression. The extension reads the entries of
#     the next power lazily (`walk_words!`), through the word tries the power was built with.
#   * Exactness. Alg. 2 is exact on the plain power, but not after the extension: the merge of the *starts* still
#     is (it drops a column, and the columns with starts are not touched by the extension), the merge of the
#     *finishes* is not (it drops a row, and the extension writes into rows with finishes, symmetrically in
#     the letters but not in the positions of the finishes). Which level of a class stands for it is then part
#     of the construction at order `τ^{N+1}`: the finishes at the back is the choice that reproduces MPSKit's
#     `TaylorCluster` to machine precision (its tuples run through the layers in reverse, so its "finishes in
#     front" is this); the other choice differs by a relative `~10^{-3}` at `τ ~ 0.3`. Everything through `τ^N` is
#     unaffected.
#   * Alg. 4 is stated for the paper's levels, where a level and its permutations are one. A level `a` with finish
#     channels and no start channel is canonical here with the finishes at the back, and its target is `a` with
#     the finishes turned into starts *and then the starts gathered at the front*, the canonical level of that
#     tuple (the column of the unsorted one was dropped by Alg. 2). Its row is discarded.
#   * Gauge. Permuting the components of a level induces `R(b_k, b_m)` per swapped pair of component charges
#     (`-1` for two fermion-odd ones). Alg. 2 only ever moves a start (or finish) component past the others,
#     and the identity channels have charge `unit`, so the gauge is trivial and the rows and columns are added
#     as they stand. (The dense operator of the merged MPO is the check: a wrong sign changes it.)
#
# Entries are `TaylorEntry`s: polynomials in `τ` with `SiteOperator` coefficients, so the construction is
# independent of `τ` and the numeric value is only taken in `evaluate` (`TaylorEntry` and `TaylorMPO` are in
# `taylormpo.jl`).

using SparseArrays: SparseMatrixCSC
using TensorKit: Sector

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
        _add!(M, r, dst, scaled(M.rows[r][src], c, k))
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
    sites::Vector{KronSite{I}}                  # the layer entries and word tries of the power, per site
    stats::PowerStats
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
rowbond(w::TaylorWork, i::Int) = i > 1 ? i - 1 : (_isfinite(w) ? 0 : nsites(w))

function taylor_work(W::SymbolicMPO{I, L}, N::Integer) where {I, L}
    P = mpo_power(W, N; closed = true)
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
        W.lattice, P.sites, P.stats, Ms, levels, [Dict(t => k for (k, t) in enumerate(lv)) for lv in levels],
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
function fold_passive!(w::TaylorWork{I, L, N}) where {I, L, N}
    removed = 0
    for j in 1:nsites(w)
        n = w.top[j]
        w.levels[j][1] == ntuple(_ -> 1, Val(N)) || invariant_violation("the start level is not first on bond $j")
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

# Alg. 2 (exact on the plain power): a level with at least as many start channels as finish channels is merged
# into the level with the starts in front, one with more finishes into the level with the finishes at the back,
# the others' order kept (see the file header for which of the row and the column is added and which dropped).
# `left = false` / `right = false` skip the merges of the starts / of the finishes. After the extension only
# the merge of the starts is exact: see the file header. Returns the number of levels removed.
function merge_equivalent!(w::TaylorWork; left::Bool = true, right::Bool = true)
    removed = 0
    for j in 1:nsites(w)
        n = w.top[j]
        M = w.Ms[j]
        nxt = _nextsite(w, j)
        for (k, t) in enumerate(w.levels[j])
            w.alive[j][k] || continue
            leftmerge = count(==(n), t) <= count(==(1), t)
            (leftmerge ? left : right) || continue
            s = leftmerge ? _gather(t, 1, true) : _gather(t, n, false)
            s == t && continue
            nxt == 0 && invariant_violation("level $t on the right boundary has to be merged: fold the passive levels first")
            target = w.index[j][s]
            leftmerge ? _addrow!(w.Ms[nxt], target, k) : _addcol!(M, target, k)
            _remove!(w, j, k)
            removed += 1
        end
    end
    return removed
end

# Alg. 3 (extension): the terms of order `N + 1` that fit the existing levels, taken from the power `H^{N+1}`,
#
#     O[a, b] += τ N! / ((N + 1)! n₁ n₃) Σ_{c,d} H^{N+1}[a with a start inserted at c, b with a finish inserted at d],
#
# `n₁` the number of starts of the first and `n₃` that of finishes of the second index (both counted after the
# insertion), for every pair of levels with no start in `b`, except for the rows that Alg. 1 removes (no start and
# a finish in `a`). The entries of `H^{N+1}` are never built as a matrix: for each `a`, `c` and `d` the layers of
# the columns `b` are enumerated over the nonzero layer entries (the finish forced in layer `d`, no start
# anywhere), and every word is multiplied once, in the tries of the power. The `N + 1` positions of an insertion
# give the same level when they are next to the same letter, so each distinct entry is read once and carries
# the multiplicity. Returns the number of entries of `H^{N+1}` read. Run before `fold_passive!` (see the file
# header).
function extend!(w::TaylorWork{I, L, N}) where {I, L, N}
    nread = 0
    for i in 1:nsites(w)
        ks = w.sites[i]
        rb = rowbond(w, i)
        n = w.top[i]
        nleft = rb == 0 ? 0 : w.top[rb]
        # the rows: the levels of the bond to the left (one level `(1, …, 1)` at the left end of a finite chain)
        rowlevels = rb == 0 ? [ntuple(_ -> 1, Val(N))] : w.levels[rb]
        for (r, a) in enumerate(rowlevels)
            rb == 0 || w.alive[rb][r] || continue
            # the rows removed by Alg. 1
            rb != 0 && all(l -> l == 1 || l == nleft, a) && any(==(nleft), a) && continue
            acc = Dict{Int, SiteOperator{I}}()
            for c in 1:(N + 1)
                # inserting the start next to another gives the same level, so the positions are taken by runs, the
                # first position of the run standing for the `1 + (length of the run of starts in a)` of them
                (c == 1 || a[c - 1] != 1) || continue
                ma = 1
                while c + ma - 1 <= N && a[c + ma - 1] == 1
                    ma += 1
                end
                ae = (a[1:(c - 1)]..., 1, a[c:N]...)
                for d in 1:(N + 1)
                    candidates(k) = if k == d
                        e = get(ks.lookup, (ae[k], n), nothing)
                        e === nothing ? () : (e,)
                    else
                        Iterators.filter(e -> e.col > 1, ks.byrow[ae[k]])
                    end
                    # likewise for the finish: only the first of a run of finishes in `b` with one inserted
                    accept(es, k) = k != d || d == 1 || es[d - 1].col != n
                    walk_words!(ks, N + 1, candidates, accept) do es, node, sgn
                        b = ntuple(j -> es[j < d ? j : j + 1].col, Val(N))
                        col = get(w.index[i], b, 0)
                        (col != 0 && w.alive[i][col]) || return
                        nread += 1
                        mb = 1
                        while d + mb <= N + 1 && es[d + mb].col == n
                            mb += 1
                        end
                        op = ks.prods[node] * (sgn * ma * mb)
                        acc[col] = haskey(acc, col) ? acc[col] + op : op
                        return
                    end
                end
            end
            n1 = count(==(1), a) + 1
            for (col, op) in acc
                n3 = count(==(n), w.levels[i][col]) + 1
                _add!(w.Ms[i], r, col, TaylorEntry(op * (factorial(N) / (factorial(N + 1) * n1 * n3)), 1))
            end
        end
    end
    return nread
end

# Alg. 4 (approximate compression): a level `a` with finish channels and no start channel is merged into the level
# with the finishes turned into starts, `O[:, b] += τ^n (N - n)! / N! · O[:, a]` for `n` finishes in `a`, and its
# row is dropped. The two agree through `τ^N`; the difference is of order `τ^{N+1}`. Returns the number of levels
# removed. Run after `merge_equivalent!` (see the file header).
function compress_levels!(w::TaylorWork{I, L, N}) where {I, L, N}
    removed = 0
    for j in 1:nsites(w)
        n = w.top[j]
        for (k, t) in enumerate(w.levels[j])
            w.alive[j][k] || continue
            a = count(==(n), t)
            (a > 0 && all(>(1), t)) || continue
            target = w.index[j][_gather(map(l -> l == n ? 1 : l, t), 1, true)]
            w.alive[j][target] || invariant_violation("the target of the compression of $t has been removed")
            _addcol!(w.Ms[j], target, k, factorial(N - a) / factorial(N), a)
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
    seed = forward ? rowbond(w, 1) : L           # the bond with a boundary level, if any
    seed == 0 || (reach[seed][1] = true)
    changed = true
    while changed
        changed = false
        for i in (forward ? (1:L) : (L:-1:1))
            M = w.Ms[i]
            src, dst = forward ? (rowbond(w, i), i) : (i, rowbond(w, i))
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
function prune_orphans!(w::TaylorWork)
    fwd, bwd = _reachable(w, true), _reachable(w, false)
    removed = 0
    for j in 1:nsites(w), k in eachindex(w.alive[j])
        w.alive[j][k] || continue
        (fwd[j][k] && bwd[j][k]) && continue
        k == 1 && invariant_violation("the start level of bond $j has no path through the chain")
        _remove!(w, j, k)
        removed += 1
    end
    return removed
end

# compact the surviving levels and finalise the matrices
function _taylor_finish(w::TaylorWork{I}, order::Int) where {I}
    newidx = [cumsum(a) for a in w.alive]
    Ws = map(1:nsites(w)) do i
        rb = rowbond(w, i)
        cb = i
        nr = rb == 0 ? 1 : count(w.alive[rb])
        dict = Dictionary{CartesianIndex{2}, TaylorEntry{I}}()
        for (r, row) in enumerate(w.Ms[i].rows), (c, e) in row
            (rb == 0 || w.alive[rb][r]) || invariant_violation("an entry survives in a removed row")
            w.alive[cb][c] || invariant_violation("an entry survives in a removed column")
            insert!(dict, CartesianIndex(rb == 0 ? 1 : newidx[rb][r], newidx[cb][c]), e)
        end
        return sparse_from_dict(dict, (nr, count(w.alive[cb])))
    end
    secs = [w.secs[j][w.alive[j]] for j in 1:nsites(w)]
    return TaylorMPO{I, typeof(w.lattice)}(w.lattice, Ws, secs, order)
end

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
    `N` times, with their exact Taylor weights. It reuses the levels of Alg. 1 + 2 but can reach levels that a
    nilpotent letter left unreachable, so the bond dimension can grow (not for the Ising chain; `16, 52, 155`
    instead of `14, 42, 117` for `N = 2, 3, 4` of the U(1) XXZ chain). Only the entries of `H^{N+1}` that are
    used are computed (never the whole power).
  - `compression` (Alg. 4): merges every level that has finish channels and no start channel into the level
    with those finishes turned into starts, with the weight `τ^n (N - n)! / N!` of its `n` finishes. This is
    *lossy* at order `τ^{N+1}` (the error stays `O(τ^{N+1})`), but combined with the extension the paper found it
    more precise than the plain construction, at a smaller bond dimension: `1 + χ`, `1 + χ + χ²`,
    `1 + 3χ + χ² + χ³`, `1 + 5χ + 4χ² + χ³ + χ⁴` for `N = 1, …, 4`.

On an [`InfiniteChain`](@ref) `H` is a generating set as in [`mpo`](@ref); the MPO is that of the unit cell.
Needs the total charge of `H` to be `unit` (like [`jordan_mpo_tensors`](@ref)) and an abelian symmetry. Every on-site product of letters is expanded in the
site's basis, which is complete, so no product leaves it.
"""
function taylor_mpo(W::SymbolicMPO, N::Integer; extension::Bool = true, compression::Bool = false)
    passes = Function[]
    extension && push!(passes, extend!)
    push!(passes, fold_passive!, merge_equivalent!)
    compression && push!(passes, compress_levels!)
    return run_passes(W, N, passes)
end
function taylor_mpo(H::OperatorSum, N::Integer; kwargs...)
    return taylor_mpo(mpo(H), N; kwargs...)
end

# The passes are `work -> count` and run in order; orphans are removed last.
function run_passes(W::SymbolicMPO, N::Integer, passes)
    work = taylor_work(W, N)
    foreach(pass -> pass(work), passes)
    prune_orphans!(work)
    return _taylor_finish(work, Int(N))
end
