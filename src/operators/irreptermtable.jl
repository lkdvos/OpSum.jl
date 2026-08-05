# Canonicalisation + the flat term table the MPO sweep consumes
# =============================================================
# A `TermList` (irrepalgebra.jl) is append-only: `couple` and `+` push columns and never look for
# coincident ones. `canonical` is the single place where that normal form is taken — coincident
# columns (identical active `(site, ITOKey)` content) accumulate their coefficients and cancelled
# terms drop out. It is memoised on the list, so the `H.terms` view, `irrep_mpo` and
# `jordan_mpo_tensors` all share one pass. (Before this, the normal form was computed *twice*: once
# by a `Dictionary{TermKey, coeff}` that `+` rebuilt from scratch, and again here.)
#
# `ITOTermTable` is the canonical list materialised as the `K × M` matrices the sweeps index into,
# plus the system size. The sweeps (irrepgraph.jl) consume nothing else; `irrep_mpo` (irrepmpo.jl)
# is the public entry.
#
# The one ITO-specific subtlety vs a dense term table: idle sites are NOT a constant identity — they
# carry a pass-through symbol with the *running* bond charge. The flat store keeps only active
# factors; `_op_at_ito` reconstructs the pass-through key at any idle position from the last active
# bond charge to its left, so the per-bond transition key is effectively `(prev_id, op, bond,
# vertex)`, preserving block-diagonality exactly.

using TensorKit: Sector, unit

# Canonicalisation
# ----------------
# A column of a term list, addressed in place. Hashing and comparing columns without materialising
# them is what keeps the dedup allocation-free per term.
struct ColumnRef{I <: Sector}
    buf::TermBuffer{I}
    t::Int
end

function Base.hash(c::ColumnRef, h::UInt)
    buf, K = c.buf, c.buf.K
    o = (c.t - 1) * K
    h = hash(:ITOColumn, h)
    @inbounds for j in 1:K
        s = buf.sites[o + j]
        iszero(s) && break
        h = hash(buf.keys[o + j], hash(s, h))
    end
    return h
end

function Base.:(==)(x::ColumnRef{I}, y::ColumnRef{I}) where {I}
    xb, yb = x.buf, y.buf
    xo, yo = (x.t - 1) * xb.K, (y.t - 1) * yb.K
    @inbounds for j in 1:max(xb.K, yb.K)
        xs = j <= xb.K ? xb.sites[xo + j] : 0
        ys = j <= yb.K ? yb.sites[yo + j] : 0
        xs == ys || return false
        iszero(xs) && return true
        xb.keys[xo + j] == yb.keys[yo + j] || return false
    end
    return true
end
Base.isequal(x::ColumnRef{I}, y::ColumnRef{I}) where {I} = x == y

"""
    canonical(H::TermSum) -> TermSum

The canonical form of `H`: coincident terms summed into one column, cancelled terms dropped, columns
in order of first appearance. Memoised on `H`, and lattice-free (the normal form does not depend on
the spaces), so binding a lattice carries the memo over unchanged.
"""
function canonical(H::TermList{I}) where {I}
    c = getfield(H, :canon)
    c === nothing || return c
    c = _canonicalize(H)
    setfield!(H, :canon, c)
    setfield!(c, :canon, c)
    return c
end

function _canonicalize(H::TermList{I}) where {I}
    buf, n, K = getfield(H, :buf), nrows(H), rowarity(H)
    # Nothing to merge for ≤ 1 column: return `H` itself when it is already exactly its buffer and
    # carries no lattice (the canonical form must stay lattice-free).
    if n <= 1 && n == buf.n && getfield(H, :lattice) === nothing &&
            (iszero(n) || !iszero(buf.coeffs[1]))
        return H
    end

    index = Dict{ColumnRef{I}, Int}()
    out = TermBuffer{I}(K)
    _sizehint!(out, n)
    sizehint!(index, n)
    for t in 1:n
        j = get(index, ColumnRef(buf, t), 0)
        if iszero(j)
            @inbounds for s in ((t - 1) * K + 1):(t * K)
                push!(out.sites, buf.sites[s])
                push!(out.keys, buf.keys[s])
            end
            push!(out.coeffs, buf.coeffs[t])
            out.n += 1
            index[ColumnRef(out, out.n)] = out.n
        else
            @inbounds out.coeffs[j] += buf.coeffs[t]
        end
    end

    any(iszero, out.coeffs) || return TermList(out, out.n)
    keep = TermBuffer{I}(K)
    _sizehint!(keep, out.n)
    @inbounds for t in 1:out.n
        iszero(out.coeffs[t]) && continue
        for s in ((t - 1) * K + 1):(t * K)
            push!(keep.sites, out.sites[s])
            push!(keep.keys, out.keys[s])
        end
        push!(keep.coeffs, out.coeffs[t])
        keep.n += 1
    end
    return TermList(keep, keep.n)
end

# The flat term table
# -------------------
"""
    ITOTermTable{I<:Sector}

Flat, sparse-per-term storage of a canonical ITO term list on an `N`-vertex chain: each term's
active `(site, ITOKey)` factors in `K×M` matrices (`sites` zero-padded, ascending) plus a parallel
`coeffs` vector. This is the only thing the reduced-MPO sweeps read.
"""
struct ITOTermTable{I <: Sector}
    sites::Matrix{Int}
    keys::Matrix{ITOKey{I}}
    coeffs::Vector{ComplexF64}
    nvertices::Int
end

arity(tt::ITOTermTable) = size(tt.sites, 1)
nterms(tt::ITOTermTable) = length(tt.coeffs)
nvertices(tt::ITOTermTable) = tt.nvertices

"""
    ITOTermTable(H::TermSum)              # `H` bound to a lattice
    ITOTermTable(H::TermSum, sites)       # binds `H` to `sites` first
    ITOTermTable(terms, sites)            # iterable of term sums, summed first

Materialise the canonical form of `H` as the flat table the MPO sweep consumes, on
`N = length(sites)` lattice sites.
"""
ITOTermTable(H::TermList) = _termtable(canonical(H), length(lattice(H)))
ITOTermTable(H::TermList, sites) = _termtable(canonical(onlattice(H, sites)), length(sites))
ITOTermTable(terms::AbstractVector{<:TermList}, sites) = ITOTermTable(sum(terms), sites)

function _termtable(H::TermList{I}, N::Int) where {I}
    M = nrows(H)
    # the *true* max arity, which the append-only stride can overshoot (a column store widened by a
    # `couple` whose widest term later cancelled). `arity(tt) ≥ 1`: the sweeps index row 1.
    K = 1
    for t in 1:M
        K = max(K, collength(H, t))
    end
    sitemat = zeros(Int, K, M)
    keymat = fill(_padkey(I), K, M)
    for t in 1:M
        for j in 1:collength(H, t)
            sitemat[j, t] = colsite(H, j, t)
            keymat[j, t] = colkey(H, j, t)
        end
    end
    return ITOTermTable{I}(sitemat, keymat, ComplexF64[colcoeff(H, t) for t in 1:M], N)
end

# ITOKey at site `s` of term `t`. On an active site it is the stored key; on an idle site it is the
# pass-through symbol carrying the running bond charge (the last active bond to the left of `s`, or
# `unit(I)` if none). Relies on `sites` columns being sorted ascending and zero-padded.
function _op_at_ito(tt::ITOTermTable{I}, t::Int, s::Int) where {I}
    lastbond = unit(I)
    @inbounds for j in 1:size(tt.sites, 1)
        st = tt.sites[j, t]
        st == 0 && break
        if st == s
            return tt.keys[j, t]
        elseif st < s
            lastbond = tt.keys[j, t].bond
        else
            break
        end
    end
    return ITOKey{I}(passthrough(I), lastbond, 1)
end
