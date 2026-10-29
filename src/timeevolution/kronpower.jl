# The N-fold Kronecker power of an MPO tensor, entirely symbolic. An entry of `W^{⊗N}` is the ordered
# on-site product of the `N` layer entries it stacks, so the power is built from structure constants
# (`site_product`) and never contracts a tensor. With the layers read as `H · H ⋯ H` the power MPO represents `H^N`.
#
# Conventions
#
#   * Level. A bond index of the power is a *level*: the tuple `(l₁, …, l_N)` of Jordan indices of the
#     bond, one per layer; its charge is the fused charge of the components (unique: abelian only).
#   * Order. Layer 1 is leftmost: `(W^{⊗N})[(l), (r)] = ± W[l₁,r₁] · W[l₂,r₂] ⋯ W[l_N,r_N]`.
#   * Sign. A layer entry is coupled `κ(bL ⊗ q → bR)`, running bond first. Stacking the layers, the letter of
#     layer `k` at site `j` passes the letters of every later layer `m > k` at the sites to its left, whose
#     total charge is the running bond charge `bL_m`: that costs `R(q_k, bL_m)`, the plain reordering of
#     letter leg `k` past bond leg `m`. On top of that, `letter_product` carries the graded sign
#     `(-1)^{p(x)p(y)}` of the placed-operator `*` for two fermion-odd letters, and `R(q_k, q_m)` is `-1`
#     exactly then, so it cancels that sign. The entry therefore carries
#     `∏_{k<m} R(q_k, bL_m) R(q_k, q_m) = ∏_{k<m} R(q_k, bR_m)`: `q_k` the letter charge of entry `k`, `bR_m`
#     the charge of the right index of entry `m`; `-1` per fermion-odd pair, `1` for a bosonic symmetry.
#     Checked against the dense `H^N` (fermionic parity, `U₁ × parity`, Hubbard); the signless entries, or
#     `R(q_k, bL_m)` alone, fail it. The equally valid `R(bL_k, q_m)` differs by the bond gauge
#     `(-1)^{p(b_k) p(b_m)}`, which is trivial at the neutral boundary indices, so it gives the same operator;
#     the rule is pairwise and multiplicative, so it composes when a power is extended by one more layer.
#
# The product of an entry is the product of its *active word*: the non-identity layer entries, in layer
# order. Layer entries are interned to small ids per site, and the partial products of all words form
# a trie, so a word is multiplied once however many level pairs carry it, and every completion of a
# prefix shares the prefix's product. The sign is the only part that depends on the bonds, and is applied
# outside the memo.

using SparseArrays: SparseMatrixCSC, sparse
using TensorKit: Sector, BraidingStyle, Bosonic, Rsymbol, dual, ⊗

# Counters of one `mpo_power`: `site_product` calls (one per distinct active word of length ≥ 2) and stored entries.
mutable struct PowerStats
    nmul::Int
    nentries::Int
end
PowerStats() = PowerStats(0, 0)

# a layer entry `W[row, col]`, interned operator id (`0` is the identity) and letter charge
struct LayerEntry{I <: Sector}
    row::Int
    col::Int
    id::Int
    charge::I
end

# `R(q, b)` for abelian symmetric braiding: ±1
_braidsign(q::I, b::I) where {I <: Sector} = Float64(real(Rsymbol(q, b, only(q ⊗ b))))

_fuse(secs) = reduce((a, b) -> only(a ⊗ b), secs)

# All levels of a bond with `n` indices, layer 1 varying fastest.
_levels(n::Int, ::Val{N}) where {N} = vec(collect(Tuple(c) for c in CartesianIndices(ntuple(_ -> n, Val(N)))))

# The layout with the right boundary of a finite chain given its start channel as well: two indices `(1, 2)` of
# charge `unit`, the start channel carrying the identity at `(1, 1)` of the last site and the old column
# moving to `2`.
function close_right(layout::JordanLayout{I}) where {I}
    N = length(layout)
    M = layout.Ws[N]
    pairs_ = collect(storedpairs(M))
    rows = Int[1; [first(Tuple(k)) for (k, _) in pairs_]]
    cols = fill(2, length(rows))
    cols[1] = 1
    vals = SiteOperator{I}[one(SiteOperator{I}); [v for (_, v) in pairs_]]
    Ws = copy(layout.Ws)
    Ws[N] = sparse(rows, cols, vals, size(M, 1), 2)
    bondsectors = copy(layout.bondsectors)
    bondsectors[N + 1] = fill(unit(I), 2)
    start, finish = copy(layout.start), copy(layout.finish)
    start[N + 1], finish[N + 1] = 1, 2
    return JordanLayout{I}(Ws, bondsectors, start, finish)
end

"""
    mpo_power(W::SymbolicMPO, N::Integer; closed = false) -> (; mpo, levels, index, stats, sites)

The `N`-fold Kronecker power of the Jordan form of `W` (`jordan_layout`): a [`SymbolicMPO`](@ref)
`mpo` whose bond indices are levels, `NTuple{N,Int}`s of Jordan indices, so that
`mpo_tensors(mpo)` contracts to `H^N` for `W` an MPO of `H` (layer 1 leftmost, see the file header).
Throws where `jordan_mpo_tensors` does, and for a non-abelian symmetry.

`levels[i]` lists the levels of the bond right of site `i` (the index of `mpo`'s bond `i` is the position
in this vector, layer 1 varying fastest), so `levels[N]` is the bond left of
site 1 on an infinite chain; the left boundary of a finite chain is the single level `(1, …, 1)`.
`index[i]` is its inverse (level ↦ position). `mpo.start`/`mpo.finish` are the positions of `(1, …, 1)` and
`(end, …, end)` (`0` where the bond has no such channel: the right boundary of a finite chain has no start). `stats` counts the work done, and `sites[i]`
is the `KronSite` of site `i`: its layer entries and word trie, from which any single entry of a power
of any order is available lazily (`walk_words!`).

`closed = true` gives the right boundary of a finite chain its start channel too (`jordan_layout`), so it has
the two identity indices `1, 2` like every other bond and `levels[N]` is all of `{1, 2}^N`: the power then
represents `H^N` for the boundary vector `(2, …, 2)`, and the other levels are what `taylor_mpo` folds.
"""
function mpo_power(W::SymbolicMPO{I}, N::Integer; closed::Bool = false) where {I}
    N >= 1 || throw(ArgumentError("the power must be at least 1, got $N"))
    checkkron(I)
    return mpo_power(W, Val(Int(N)), closed)
end

function mpo_power(W::SymbolicMPO{I}, ::Val{N}, closed) where {I, N}
    layout = jordan_layout(W)
    closed && W.lattice isa FiniteChain && (layout = close_right(layout))
    L = length(W)
    # bond `b` is left of site `b`; the right bond of an infinite chain is the first again
    levels = Vector{Vector{NTuple{N, Int}}}(undef, L + 1)
    secs = Vector{Vector{I}}(undef, L + 1)
    for b in 1:(L + 1)
        levels[b] = _levels(length(layout.bondsectors[b]), Val(N))
        secs[b] = [_fuse(layout.bondsectors[b][collect(t)]) for t in levels[b]]
    end
    index = [Dict(t => k for (k, t) in enumerate(lv)) for lv in levels]

    stats = PowerStats()
    built = map(1:L) do i
        return _power_site(
            W.lattice[i], layout.Ws[i], Val(N), index[i], index[i + 1],
            layout.bondsectors[i], layout.bondsectors[i + 1], stats
        )
    end
    sites, Ws = first.(built), last.(built)

    first_ = ntuple(_ -> 1, Val(N))
    last_(b) = ntuple(_ -> length(layout.bondsectors[b]), Val(N))
    pick(b, has, t) = has ? get(index[b], t, 0) : 0
    start = [pick(b, !iszero(layout.start[b]), first_) for b in 2:(L + 1)]
    finish = [pick(b, !iszero(layout.finish[b]), last_(b)) for b in 2:(L + 1)]
    mpo = SymbolicMPO{I, typeof(W.lattice)}(W.lattice, Ws, secs[2:(L + 1)], start, finish)
    return (; mpo, levels = levels[2:(L + 1)], index = index[2:(L + 1)], stats, sites)
end

# One site of a power: the layer entries `W[l, r]` with interned operators, and the trie of active words. A word is
# multiplied once, however many level pairs carry it (and whatever the number of layers: the trie is the same for
# a power of any order, which is what lets `extend!` evaluate entries of the next power lazily). `node 1` is the
# empty word; `children[(node, id)]` is the node of the word extended by entry `id`, `0` if the product vanishes.
struct KronSite{I <: Sector}
    basis::Basis{I}
    ops::Vector{SiteOperator{I}}
    entries::Vector{LayerEntry{I}}
    lookup::Dict{Tuple{Int, Int}, LayerEntry{I}}
    byrow::Vector{Vector{LayerEntry{I}}}
    secR::Vector{I}
    prods::Vector{SiteOperator{I}}
    children::Dict{Tuple{Int, Int}, Int}
    stats::PowerStats
end

function KronSite(
        B::Basis{I}, Wi::SparseMatrixCSC{SiteOperator{I}}, secL::Vector{I}, secR::Vector{I}, stats::PowerStats
    ) where {I}
    ops = SiteOperator{I}[]
    ids = Dict{SiteOperator{I}, Int}()
    entries = LayerEntry{I}[]
    for (idx, op) in storedpairs(Wi)
        l, r = Tuple(idx)
        id = isone(op) ? 0 : get!(ids, op) do
                push!(ops, op)
                return length(ops)
        end
        push!(entries, LayerEntry{I}(l, r, id, only(secR[r] ⊗ dual(secL[l]))))
    end
    lookup = Dict((e.row, e.col) => e for e in entries)
    byrow = [LayerEntry{I}[] for _ in 1:size(Wi, 1)]
    foreach(e -> push!(byrow[e.row], e), entries)
    return KronSite{I}(
        B, ops, entries, lookup, byrow, secR, SiteOperator{I}[one(SiteOperator{I})],
        Dict{Tuple{Int, Int}, Int}(), stats
    )
end

# the node of the word of `node` extended by the layer entry `id`; a node's product is only ever built here
function _extend_word!(ks::KronSite, node::Int, id::Int)
    return get!(ks.children, (node, id)) do
        if node == 1
            push!(ks.prods, ks.ops[id])
        else
            ks.stats.nmul += 1
            P = site_product(ks.basis, ks.prods[node], ks.ops[id])
            isempty(P) && return 0
            push!(ks.prods, P)
        end
        return length(ks.prods)
    end
end

# Depth-first over the `n` layers of an entry, so that the partial product of layers `1..k` is shared by all
# completions: layer `k` tries the entries of `candidates(k)` that `accept(es, k)` (given the entries `es[1:k]`
# chosen so far), and `leaf(es, node, sgn)` is called with every complete choice, the node of its product and its
# braiding sign (see the file header).
function walk_words!(leaf, ks::KronSite{I}, n::Int, candidates, accept) where {I}
    braided = !(BraidingStyle(I) isa Bosonic)
    es = Vector{LayerEntry{I}}(undef, n)
    function visit(k::Int, node::Int, sgn::Float64)
        if k > n
            leaf(es, node, sgn)
            return
        end
        for e in candidates(k)
            es[k] = e
            accept(es, k) || continue
            child = e.id == 0 ? node : _extend_word!(ks, node, e.id)
            iszero(child) && continue
            s = sgn
            if braided
                for j in 1:(k - 1)
                    s *= _braidsign(es[j].charge, ks.secR[e.col])
                end
            end
            visit(k + 1, child, s)
        end
        return
    end
    visit(1, 1, 1.0)
    return
end

# One site of the power: all nonzero entries, depth-first over the layers.
function _power_site(
        B::Basis{I}, Wi::SparseMatrixCSC{SiteOperator{I}}, ::Val{N}, rowindex, colindex,
        secL::Vector{I}, secR::Vector{I}, stats::PowerStats
    ) where {I, N}
    ks = KronSite(B, Wi, secL, secR, stats)
    rows, cols = Int[], Int[]
    vals = SiteOperator{I}[]
    walk_words!(ks, N, _ -> ks.entries, (_, _) -> true) do es, node, sgn
        push!(rows, rowindex[ntuple(j -> es[j].row, Val(N))])
        push!(cols, colindex[ntuple(j -> es[j].col, Val(N))])
        push!(vals, sgn == 1 ? ks.prods[node] : ks.prods[node] * sgn)
        return
    end
    stats.nentries += length(vals)
    return ks, sparse(rows, cols, vals, length(rowindex), length(colindex))
end
