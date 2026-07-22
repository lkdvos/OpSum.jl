@doc """
    TermTable{Op,T}

Flat, sparse-per-term storage of a sum of operator strings on an `N`-vertex
chain. This is the flat replacement for the pointer-based [`Trie`](@ref) as the
term list feeding MPO bond optimization, mirroring the sparse `(site, op)`-pair
encoding of ITensorMPOConstruction's `OpIDSum`.

Each term is a product of local operators. Only its **non-identity**
`(site, op)` factors are stored (identities are implicit), sorted by `site` and
padded to the table's maximum arity `K` with the sentinel site `0`. A parallel
`coeffs` vector holds each term's scalar coefficient.

Fields:
- `sites::Matrix{Int}`: `K × M`; column `t` lists term `t`'s occupied sites in
  ascending order, zero-padded in unused slots.
- `ops::Matrix{Op}`: `K × M`; `ops[k, t]` is the operator at site `sites[k, t]`
  (padded with `one(Op)` where `sites[k, t] == 0`).
- `coeffs::Vector{T}`: length `M`; the coefficient of each term.
- `nvertices::Int`: number of chain vertices `N`.

The columns are canonical: two terms with the same set of non-identity factors
occupy a single column with their coefficients summed, matching the
insertion-time duplicate summation of `build_trie!`.
""" TermTable

struct TermTable{Op, T}
    sites::Matrix{Int}
    ops::Matrix{Op}
    coeffs::Vector{T}
    nvertices::Int
end

# Properties
# ----------
"""Maximum term arity `K` (number of stored `(site, op)` slots per term)."""
arity(tt::TermTable) = size(tt.sites, 1)

"""Number of (canonical) terms `M`."""
nterms(tt::TermTable) = length(tt.coeffs)

"""Number of chain vertices `N`."""
nvertices(tt::TermTable) = tt.nvertices

Base.eltype(::Type{TermTable{Op, T}}) where {Op, T} = Pair{Vector{Pair{Int, Op}}, T}

# Construction from a GlobalOp
# ----------------------------
"""
    TermTable(vertices, ex::GlobalOp)

Build a [`TermTable`](@ref) directly from a `GlobalOp` expression on the chain
`vertices`, expanding `Sum`/`SiteOp` terms into canonical sparse operator
strings. This replaces `build_trie!`'s role: no `Trie` is ever materialised.

Exact-duplicate terms (identical sets of non-identity `(site, op)` factors) are
merged and their coefficients summed at construction time, matching the
insertion-time summation performed by `build_trie!`/`_emit_leaf!`.
"""
function TermTable(vertices, ex::GlobalOp{T, A, S}) where {T, A, S}
    N = length(vertices)

    # Canonical sparse content -> row index, for insertion-time deduplication.
    index = Dictionary{Vector{Pair{Int, A}}, Int}()
    keys_ = Vector{Pair{Int, A}}[]
    coeffs = T[]

    _collect_terms!(index, keys_, coeffs, vertices, ex, one(T))

    M = length(keys_)
    K = max(2, maximum(length, keys_; init = 0))

    sites = zeros(Int, K, M)
    ops = fill(one(A), K, M)
    for t in 1:M
        term = keys_[t]
        for (j, (s, o)) in enumerate(term)
            sites[j, t] = s
            ops[j, t] = o
        end
    end

    return TermTable{A, T}(sites, ops, coeffs, N)
end

# Recursively walk the expression tree, mirroring `build_trie!`
# (globalalgebra.jl), but emitting sparse `(site, op)` rows instead of dense
# length-`N` trie keys.
function _collect_terms!(
        index, keys_, coeffs, vertices, ex::GlobalOp{T, A, S}, coeff::T
    ) where {T, A, S}
    iszero(coeff) && return nothing
    o = variant(ex)

    if o isa Sum
        for (k, v) in pairs(o.terms)
            _collect_terms!(index, keys_, coeffs, vertices, k, coeff * v)
        end

    elseif o isa SiteOp
        @assert issorted(o.sites) && allunique(o.sites)
        local_coeffs, local_ops = operatorstrings(o.op)

        if isempty(o.sites)
            # Constant/identity term: no non-identity factors.
            for (lc, _lop) in zip(local_coeffs, local_ops)
                iszero(lc) && continue
                _emit_term!(index, keys_, coeffs, (), coeff * lc)
            end
        elseif length(o.sites) == 1
            site_pos = only(o.sites)
            for (lc, lop) in zip(local_coeffs, local_ops)
                iszero(lc) && continue
                _emit_term!(index, keys_, coeffs, ((site_pos, lop),), coeff * lc)
            end
        else
            for (lc, lop) in zip(local_coeffs, local_ops)
                iszero(lc) && continue
                _emit_term!(index, keys_, coeffs, zip(o.sites, lop), coeff * lc)
            end
        end

    else
        error("TermTable: unsupported GlobalOp variant $(typeof(o))")
    end
    return nothing
end

# Canonicalise a term's raw `(site, op)` pairs (drop identities, sort by site)
# and insert it, summing coefficients on an exact-duplicate content match.
function _emit_term!(index, keys_, coeffs::Vector{T}, rawpairs, coeff::T) where {T}
    key = eltype(keys_)()
    for (s, o) in rawpairs
        isone(o) && continue
        push!(key, s => o)
    end
    sort!(key; by = first)

    tok = get(index, key, 0)
    if iszero(tok)
        push!(keys_, key)
        push!(coeffs, coeff)
        insert!(index, key, length(keys_))
    else
        coeffs[tok] += coeff
    end
    return nothing
end

# Iteration: yield `sparse_key => coeff` for each stored term, where
# `sparse_key` is the sorted vector of non-identity `(site => op)` pairs.
function Base.iterate(tt::TermTable{Op, T}, t::Int = 1) where {Op, T}
    t > nterms(tt) && return nothing
    key = Pair{Int, Op}[]
    for j in 1:arity(tt)
        s = tt.sites[j, t]
        s == 0 && continue
        push!(key, s => tt.ops[j, t])
    end
    return (key => tt.coeffs[t], t + 1)
end
Base.length(tt::TermTable) = nterms(tt)
Base.IteratorSize(::Type{<:TermTable}) = Base.HasLength()

# Show
# ----
function Base.show(io::IO, ::MIME"text/plain", tt::TermTable{Op, T}) where {Op, T}
    M = nterms(tt)
    print(io, TermTable{Op, T}, " (", M, M == 1 ? " term" : " terms", ", N=", nvertices(tt), ", K=", arity(tt), ")")
    for (key, c) in tt
        print(io, "\n  ")
        show(io, c)
        print(io, " * ")
        if isempty(key)
            print(io, "𝟙")
        else
            join(io, ("$(o)@$(s)" for (s, o) in key), " ")
        end
    end
    return nothing
end
