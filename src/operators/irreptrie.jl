# Charge-augmented automaton (redesign): trie derived from the term list
# ======================================================================
# Turns a `TermSum` (sum of sparse, sited fusion-resolved terms) into a charge-augmented `Trie`
# whose alphabet symbol is `(ITO, bond_charge_out, vertex)`. The trie is a *derived index* over
# the term list — it makes the automaton block-diagonal in the bond charge, which is what Phase 4
# (per-sector bipartite matching + SVD) consumes.
#
# Design notes:
# * Idle sites are filled with a distinguished pass-through identity symbol (trivial charge, index
#   `n = 0`), not an enumerated `(c, n)` letter and not `one(A)`; it acts as `id(V)`.
# * The coupling structure (per-site bond charges + vertex labels) is derived from each term's
#   `(charges, total)` via TensorKit `fusiontrees` (bounded, canonical) and lives entirely in the
#   trie *key*; the leaf value stays a single scalar reduced coefficient.
# * Supported term arity K ∈ {0,1,2}; multi-channel (GenericFusion) / multi-body deferred.

using TensorKit
using TensorKit: Sector, FusionTree, fusiontrees, unit
using .IrrepTensorOperators: IrrepOperator

# Pass-through identity symbol
# ----------------------------
"""
    passthrough(::Type{I}) where {I<:Sector}

The distinguished structural pass-through identity symbol: trivial charge `unit(I)` with the
sentinel index `n = 0`. Fills idle sites, injecting trivial charge and acting as `id(V)`. Not one
of the enumerated `(c, n)` alphabet letters (`n ≥ 1`) and not a type-level `one`.
"""
passthrough(::Type{I}) where {I <: Sector} = IrrepOperator{I}(unit(I), 0)

"""
    ispassthrough(op::IrrepOperator)

Whether `op` is the pass-through identity sentinel (index `n = 0`).
"""
ispassthrough(op::IrrepOperator) = op.n == 0

# Augmented trie key
# ------------------
"""
    ITOKey{I<:Sector}

Alphabet symbol of the charge-augmented automaton: `op` (the ITO letter or pass-through), `bond`
(the bond charge *out* of this site, i.e. the running fusion up to and including it), and `vertex`
(the fusion-vertex multiplicity label; `1` for multiplicity-free fusion). Prefixes agreeing on op
*and* bond *and* vertex share a trie node, so the trie is a direct sum of per-bond-sector FSMs.
"""
struct ITOKey{I <: Sector}
    op::IrrepOperator{I}
    bond::I
    vertex::Int
end

function Base.isless(x::ITOKey{I}, y::ITOKey{I}) where {I}
    x.op == y.op || return isless(x.op, y.op)
    x.bond == y.bond || return isless(x.bond, y.bond)
    return isless(x.vertex, y.vertex)
end
Base.:(==)(x::ITOKey{I}, y::ITOKey{I}) where {I} =
    x.op == y.op && x.bond == y.bond && x.vertex == y.vertex
Base.hash(x::ITOKey, h::UInt) = hash(x.vertex, hash(x.bond, hash(x.op, hash(:ITOKey, h))))

function Base.show(io::IO, k::ITOKey)
    return print(io, "ITOKey(", k.op, ", bond=", k.bond, ", μ=", k.vertex, ")")
end

# Caterpillar coupling tree + its per-site charges/labels
# -------------------------------------------------------
"""
    caterpillar_trees(charges::NTuple{K,I}, total::I) -> Vector{FusionTree{I,K}}

All left-nested caterpillar fusion trees of the per-site `charges` fusing to `total`, in
TensorKit's canonical order (finite, bounded — delegated to TensorKit `fusiontrees`).
"""
function caterpillar_trees(charges::NTuple{K, I}, total::I) where {K, I <: Sector}
    K == 0 && throw(ArgumentError("empty charge tuple"))
    return collect(fusiontrees(charges, total, ntuple(_ -> false, K)))
end

"""
    bondcharges(f::FusionTree{I,K})

Running bond charges `b₁…b_K` of the caterpillar tree (`b₁ = uncoupled[1]`, `bₖ = innerlines[k-1]`
for `2 ≤ k ≤ K-1`, `b_K = coupled`); element `k` is the bond charge *out* of active position `k`.
"""
function bondcharges(f::FusionTree{I, K}) where {I, K}
    K == 0 && return I[]
    K == 1 && return I[f.coupled]
    return I[f.uncoupled[1], f.innerlines..., f.coupled]
end

_vlabel(v) = v isa Integer ? Int(v) : 1

"""
    vertexlabels(f::FusionTree{I,K})

Fusion-vertex multiplicity labels, one per active position (`1` at position 1, `f.vertices[k-1]`
after).
"""
function vertexlabels(f::FusionTree{I, K}) where {I, K}
    K == 0 && return Int[]
    return Int[k == 1 ? 1 : _vlabel(f.vertices[k - 1]) for k in 1:K]
end

# Expand a sparse term into the full per-lattice-site `ITOKey` path (pass-through on idle sites,
# with the running bond charge threaded through).
function _ito_path(tk::TermKey{I, S}, N::Integer) where {I, S}
    path = Vector{ITOKey{I}}(undef, N)
    K = length(tk.sites)
    if K == 0
        for k in 1:N
            path[k] = ITOKey{I}(passthrough(I), unit(I), 1)
        end
        return path
    end
    charges = ntuple(i -> tk.ops[i].c, K)
    trees = caterpillar_trees(charges, tk.total)
    @assert length(trees) == 1 "expected a unique coupling channel (multi-channel deferred), got $(length(trees))"
    tree = only(trees)
    bonds = bondcharges(tree)      # K values (per active position)
    verts = vertexlabels(tree)     # K values
    ai = 1                          # next active index
    seen = 0                        # active sites seen so far
    for k in 1:N
        if ai <= K && tk.sites[ai] == k
            path[k] = ITOKey{I}(tk.ops[ai], bonds[ai], verts[ai])
            seen = ai
            ai += 1
        else
            bond = seen == 0 ? unit(I) : bonds[seen]
            path[k] = ITOKey{I}(passthrough(I), bond, 1)
        end
    end
    return path
end

# Charge-augmented trie
# ---------------------
"""
    irrep_trie(ts::TermSum, sites) -> Trie{ITOKey{I}, ComplexF64}
    irrep_trie(terms, sites)       # iterable of TermSums, summed first

Build the charge-augmented automaton trie for a term-sum over `N = length(sites)` lattice sites.
Each term contributes its `ITOKey` path (pass-through on idle sites); coincident paths accumulate
coefficients. The trie is block-diagonal in `ITOKey.bond`.
"""
function irrep_trie(ts::TermSum{I, S, T}, sites) where {I, S, T}
    N = length(sites)
    K = ITOKey{I}
    root = Trie{K, ComplexF64}()
    for (termkey, coeff) in pairs(ts.terms)
        path = _ito_path(termkey, N)
        node = root
        for k in path
            node = haskey(node.children, k) ? node.children[k] : _add_child!(node, k)
        end
        node.value = isnothing(node.value) ? ComplexF64(coeff) : node.value + ComplexF64(coeff)
    end
    return sortkeys!(root)
end

irrep_trie(terms::AbstractVector{<:TermSum}, sites) = irrep_trie(reduce(+, terms), sites)

# Round-trip: reconstruct the term list from trie paths
# -----------------------------------------------------
"""
    trie_terms(trie::Trie{ITOKey{I}, T}) -> TermSum{I, Int, T}

Reconstruct the (sparse) term-sum from the paths of a charge-augmented `trie` (inverse of
`irrep_trie` up to ordering and coefficient accumulation). Active sites are the non-pass-through
positions; the total charge is the bond out of the last site.
"""
function trie_terms(trie::Trie{ITOKey{I}, T}) where {I, T}
    d = Dictionary{TermKey{I, Int}, T}()
    for (path, coeff) in pairs(trie)
        sites = Int[]
        ops = IrrepOperator{I}[]
        for (k, key) in enumerate(path)
            if !ispassthrough(key.op)
                push!(sites, k)
                push!(ops, key.op)
            end
        end
        total = last(path).bond
        setwith!(+, d, TermKey{I, Int}(sites, ops, total), coeff)
    end
    return TermSum{I, Int, T}(d)
end
