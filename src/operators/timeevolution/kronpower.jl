# The N-fold Kronecker power of an MPO tensor, entirely symbolic. An entry of `W^{⊗N}` is the ordered
# on-site product of the `N` layer entries it stacks, so the power is built from structure constants
# (`_mul`) and never contracts a tensor. With the layers read as `H · H ⋯ H` the power MPO represents `H^N`.
#
# Conventions
#
#   * Level. A bond index of the power is a *level*: the tuple `(l₁, …, l_N)` of Jordan indices of the
#     bond, one per layer; its charge is the fused charge of the components (unique: abelian only).
#   * Order. Layer 1 is leftmost: `(W^{⊗N})[(l), (r)] = ± W[l₁,r₁] · W[l₂,r₂] ⋯ W[l_N,r_N]`.
#   * Sign. A layer entry is coupled `κ(bL ⊗ q → bR)`, running bond first. Stacking the layers, the letter of
#     layer `k` at site `j` has to pass the letters of every later layer `m > k` at the sites left of `j`
#     (whose total charge is the running bond charge `bL_m`): `R(q_k, bL_m)`. On top of that each site
#     needs `R(q_k, q_m)`, the order of the two charge legs that `_letter_product` does not account for
#     (found against the dense oracle, not derived). The net braiding is `R(q_k, bL_m) R(q_k, q_m) = R(q_k, bR_m)`, so the entry carries
#     `∏_{k<m} R(q_k, bR_m)`: `q_k` the letter charge of entry `k`, `bR_m` the charge of the right index of
#     entry `m`; `-1` per fermion-odd pair, `1` for a bosonic symmetry. Checked against the dense `H^N`
#     (fermionic parity, `U₁ × parity`, Hubbard); the signless entries, or `R(q_k, bL_m)` alone, fail it.
#     The equally valid `R(bL_k, q_m)` differs by the bond gauge `(-1)^{p(b_k) p(b_m)}`, which is trivial at
#     the neutral boundary indices, so it gives the same operator; the rule is pairwise and multiplicative,
#     so it composes when a power is extended by one more layer.
#
# The product of an entry is the product of its *active word*: the non-identity layer entries, in layer
# order. Layer entries are interned to small ids per site, and the partial products of all words form
# a trie, so a word is multiplied once however many level pairs carry it, and every completion of a
# prefix shares the prefix's product. The sign is the only part that depends on the bonds, and is applied
# outside the memo.

using SparseArrays: SparseMatrixCSC, sparse
using TensorKit: Sector, BraidingStyle, Bosonic, Rsymbol, dual, ⊗

# Counters of one `_mpo_power`: `_mul` calls, distinct active words of length ≥ 2 (one `_mul` each) and stored entries.
mutable struct PowerStats
    nmul::Int
    nwords::Int
    nentries::Int
end
PowerStats() = PowerStats(0, 0, 0)

# Hook for pruning columns: `keep(prefix)` is asked about every prefix `(l₁, …, l_k)` of a level, and a level
# is kept if all its prefixes are. The default keeps everything.
_keepall(::Tuple) = true

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

# All levels of a bond with `n` indices whose prefixes `keep` accepts, layer 1 varying fastest.
function _levels(n::Int, ::Val{N}, keep) where {N}
    tuples = (Tuple(c) for c in CartesianIndices(ntuple(_ -> n, Val(N))))
    keep === _keepall && return vec(collect(tuples))
    return [t for t in tuples if all(k -> keep(t[1:k]), 1:N)]
end

"""
    _mpo_power(W::SymbolicMPO, N::Integer; keep = _keepall) -> (; mpo, levels, stats)

The `N`-fold Kronecker power of the Jordan form of `W` (`_jordan_layout`): a [`SymbolicMPO`](@ref)
`mpo` whose bond indices are levels, `NTuple{N,Int}`s of Jordan indices, so that
`irrep_mpo_tensors(mpo)` contracts to `H^N` for `W` an MPO of `H` (layer 1 leftmost, see the file header).
Throws where `jordan_mpo_tensors` does, and for a non-abelian symmetry.

`levels[i]` lists the levels of the bond right of site `i` (the index of `mpo`'s bond `i` is the position
in this vector, layer 1 varying fastest unless `keep` removed some), so `levels[N]` is the bond left of
site 1 on an infinite chain; the left boundary of a finite chain is the single level `(1, …, 1)`.
`mpo.start`/`mpo.finish` are the positions of `(1, …, 1)` and `(end, …, end)` (`0` where the bond has no
such channel: the right boundary of a finite chain has no start). `stats` counts the work done.

`keep(prefix::Tuple)` is the hook for pruning: it is asked about every prefix of every level and a level
survives only if all prefixes do.
"""
function _mpo_power(W::SymbolicMPO{I}, N::Integer; keep = _keepall) where {I}
    N >= 1 || throw(ArgumentError("the power must be at least 1, got $N"))
    _checkkron(I)
    return _mpo_power(W, Val(Int(N)), keep)
end

function _mpo_power(W::SymbolicMPO{I}, ::Val{N}, keep) where {I, N}
    layout = _jordan_layout(W)
    L = length(W)
    # bond `b` is left of site `b`; the right bond of an infinite chain is the first again
    levels = Vector{Vector{NTuple{N, Int}}}(undef, L + 1)
    secs = Vector{Vector{I}}(undef, L + 1)
    for b in 1:(L + 1)
        levels[b] = _levels(length(layout.bondsectors[b]), Val(N), keep)
        secs[b] = [_fuse(layout.bondsectors[b][collect(t)]) for t in levels[b]]
    end
    index = [Dict(t => k for (k, t) in enumerate(lv)) for lv in levels]

    stats = PowerStats()
    Ws = map(1:L) do i
        return _power_site(
            W.lattice[i], layout.Ws[i], Val(N), index[i], index[i + 1],
            layout.bondsectors[i], layout.bondsectors[i + 1], keep, stats
        )
    end

    first_ = ntuple(_ -> 1, Val(N))
    last_(b) = ntuple(_ -> length(layout.bondsectors[b]), Val(N))
    pick(b, has, t) = has ? get(index[b], t, 0) : 0
    start = [pick(b, !iszero(layout.start[b]), first_) for b in 2:(L + 1)]
    finish = [pick(b, !iszero(layout.finish[b]), last_(b)) for b in 2:(L + 1)]
    mpo = SymbolicMPO(W.lattice, Ws, secs[2:(L + 1)], start, finish)
    return (; mpo, levels = levels[2:(L + 1)], stats)
end

# One site: depth-first over the layers, so that the partial product of layers `1..k` is shared by all
# completions, with the trie of active words memoising the products across the whole site.
function _power_site(
        B::Basis{I}, Wi::SparseMatrixCSC{SiteOperator{I}}, ::Val{N}, rowindex, colindex,
        secL::Vector{I}, secR::Vector{I}, keep, stats::PowerStats
    ) where {I, N}
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

    # the trie: node 1 is the empty word; `children[(node, id)]` is the node of the word extended by
    # entry `id`, `0` if the product vanishes. A node's product is only ever built here.
    prods = SiteOperator{I}[one(SiteOperator{I})]
    children = Dict{Tuple{Int, Int}, Int}()
    function extend(node::Int, id::Int)
        return get!(children, (node, id)) do
            if node == 1
                push!(prods, ops[id])
            else
                stats.nmul += 1
                P = _mul(B, prods[node], ops[id])
                isempty(P) && return 0
                push!(prods, P)
            end
            return length(prods)
        end
    end

    braided = !(BraidingStyle(I) isa Bosonic)
    ls, rs = zeros(Int, N), zeros(Int, N)
    qs = Vector{I}(undef, N)
    rows, cols = Int[], Int[]
    vals = SiteOperator{I}[]
    accepted(v, k) = keep === _keepall || keep(ntuple(j -> v[j], k))

    function visit(k::Int, node::Int, sgn::Float64)
        if k > N
            push!(rows, rowindex[ntuple(j -> ls[j], Val(N))])
            push!(cols, colindex[ntuple(j -> rs[j], Val(N))])
            push!(vals, sgn == 1 ? prods[node] : prods[node] * sgn)
            return
        end
        for e in entries
            ls[k], rs[k] = e.row, e.col
            (accepted(ls, k) && accepted(rs, k)) || continue
            child = e.id == 0 ? node : extend(node, e.id)
            iszero(child) && continue
            s = sgn
            if braided
                for j in 1:(k - 1)
                    s *= _braidsign(qs[j], secR[e.col])
                end
            end
            qs[k] = e.charge
            visit(k + 1, child, s)
        end
        return
    end
    visit(1, 1, 1.0)

    stats.nwords += count(key -> key[1] != 1, keys(children))
    stats.nentries += length(vals)
    return sparse(rows, cols, vals, length(rowindex), length(colindex))
end
