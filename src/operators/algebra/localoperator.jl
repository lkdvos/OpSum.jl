# Unplaced multi-site operators. A `Term` is `(sites, keys, coeff)`, so until now a K-site operator
# could only exist *placed*; `project(h, sites)` had to take the sites and every model re-projected
# the same bond block once per bond. `LocalOperator{I}` fills that cell of the type grid: a `Terms`
# bag on the relative sites `1:K`, placed by `B[i]` / `B[s₁, …, s_K]` — which is a pure relabelling
# of each term's `sites`, so the keys (letters, running bond charges, vertex labels) carry over
# untouched and `canonicalize!`, `≈`, `==` and `show` are reused unchanged.
#
# `SiteOperator` stays a separate type: it is the entry type of every bond matrix in the sweep and
# must remain a flat letter → coefficient map. "Site" means one site, "local" means K.
#
# Design note: `research/local-operators.md` §3. Unplaced `couple`/`dot` and passthrough slots (§4)
# and the lattice-carrying container (§5) are later steps.

using TensorKit: Sector
using VectorInterface: VectorInterface, scale

"""
    LocalOperator{I<:Sector}

An operator on `K` consecutive **slots**, not yet placed on the lattice: a [`Terms`](@ref) bag on the
relative sites `1:K`, every term active on all `K` of them. Obtained from [`project`](@ref) (the
`project(h)` form) or by converting a [`SiteOperator`](@ref) (`K = 1`), and combined with ordinary
arithmetic (`+ - * /`, between operators of equal `K`).

Place it with indexing, which produces a `Terms` bag and is where the slots become lattice sites:

```julia
B = project(h_bond)          # K = 2
B[i]                         # contiguous: sites i:i+K-1
B[i, j]                      # explicit: K strictly increasing sites; the gap is passed through
```

Placement relabels the sites and nothing else — keys and coefficients are shared with `B` — so it is
valid for any symmetry, including fermionic ones: a site inside a gap is a pass-through, and the
sweep reconstructs the running bond charge (and with it the graded sign) crossing it.

`length(B)` is the number of canonical terms, as for `Terms`; `nsites(B)` is `K`, which the
operator remembers even when it has no terms (`zero(B)`).
"""
struct LocalOperator{I <: Sector}
    terms::Terms{I}
    nsites::Int
    function LocalOperator{I}(terms::Terms{I}, nsites::Integer) where {I <: Sector}
        K = Int(nsites)
        K >= 1 || throw(ArgumentError("a LocalOperator must act on at least one slot, got nsites = $K"))
        # Full support is the invariant everything here leans on: placement maps slot `k` to
        # `sites[k]`, which has no meaning for a term that skips a slot.
        for t in terms.terms
            t.sites == 1:K || throw(
                ArgumentError(
                    "every term of a LocalOperator on $K slots must act on exactly the slots 1:$K, " *
                        "got a term on sites $(t.sites)"
                )
            )
        end
        return new{I}(terms, K)
    end
end

# Constructors
# ------------
LocalOperator(terms::Terms{I}, nsites::Integer) where {I} = LocalOperator{I}(terms, nsites)

# `K` read off the terms; an empty bag cannot say how many slots it has.
function LocalOperator(terms::Terms{I}) where {I}
    isempty(terms.terms) && throw(
        ArgumentError("cannot infer the number of slots of an empty LocalOperator; pass `nsites`")
    )
    return LocalOperator{I}(terms, arity(first(terms.terms)))
end

"""
    LocalOperator(O::SiteOperator) -> LocalOperator

The one-slot operator with the letters of `O`. A `SiteOperator` with a pass-through (identity)
component has no `LocalOperator` form yet, and is an error.
"""
function LocalOperator(O::SiteOperator{I}) where {I}
    any(ispassthrough, keys(O)) && throw(
        ArgumentError(
            "a scalar (pass-through) part is not yet supported in a LocalOperator; only charged " *
                "and trivial-charge *letters* can occupy a slot"
        )
    )
    return LocalOperator{I}(O[1], 1)
end

sectortype(::Type{LocalOperator{I}}) where {I} = I
VectorInterface.scalartype(::Type{<:LocalOperator}) = ComplexF64

"""
    nsites(B::LocalOperator) -> Int

The number of slots `K` of `B`, i.e. the number of sites every placement `B[…]` acts on.
"""
nsites(B::LocalOperator) = B.nsites

# Container interface: the canonical term set, exactly as `Terms` reports it.
Base.length(B::LocalOperator) = length(B.terms)
Base.isempty(B::LocalOperator) = isempty(B.terms)
Base.iszero(B::LocalOperator) = isempty(B.terms)
Base.iterate(B::LocalOperator, args...) = iterate(B.terms, args...)
Base.eltype(::Type{LocalOperator{I}}) where {I} = Term{I}

# Placement
# ---------
"""
    B[i]               -> Terms      # contiguous: sites i:i+K-1
    B[s₁, s₂, …, s_K]  -> Terms      # explicit, strictly increasing

Place a `K`-slot [`LocalOperator`](@ref) on the lattice. One index places it contiguously from site
`i`; `K` indices name every site, in strictly increasing order (for `K = 1` the two readings
coincide). Any other number of indices is an error.

Placement is the monotone relabelling `k ↦ sites[k]` of each term's sites: letters, running bond
charges and coefficients are unchanged, so it is exact for any symmetry. Sites inside a gap become
pass-throughs, which the sweep reconstructs — for a fermionic bond charge crossing the gap this is
what carries the graded (Jordan–Wigner) string, so `project(hop)[i, j]` is exactly the term
`couple(cd[i], c[j])` writes.
Non-monotone placement (`B[j, i]` with `j > i`) would need braiding data and is not supported.

```jldoctest
julia> using TensorKit, LinearAlgebra

julia> V = SU2Space(1//2 => 1);

julia> B = project(OpSum.instantiate(dot(spin(V)[1], spin(V)[2]), [V, V]));

julia> B[3] ≈ dot(spin(V)[3], spin(V)[4])
true

julia> B[1, 4] ≈ dot(spin(V)[1], spin(V)[4])
true
```
"""
function Base.getindex(B::LocalOperator{I}, ind::Integer, inds::Integer...) where {I}
    K = B.nsites
    n = 1 + length(inds)
    if n == 1
        first = Int(ind)
        first >= 1 || throw(ArgumentError("site index must be ≥ 1, got $first"))
        sites = collect(first:(first + K - 1))
    elseif n == K
        sites = Int[Int(ind), (Int(s) for s in inds)...]
        sites[1] >= 1 || throw(ArgumentError("site index must be ≥ 1, got $(sites[1])"))
        all(k -> sites[k] < sites[k + 1], 1:(K - 1)) || throw(
            ArgumentError("placement sites must be strictly increasing, got $sites")
        )
    else
        throw(
            ArgumentError(
                "a LocalOperator on $K slots is placed with one site index (contiguous) or exactly " *
                    "$K (explicit), got $n"
            )
        )
    end
    # Canonicalise once here rather than once per placed copy: the bag is the same operator either
    # way, and a translation-invariant model places the same block `N` times.
    ts = canonicalize!(B.terms).terms
    # Terms are never mutated, so every placed term may share the one `sites` vector.
    out = Term{I}[Term{I}(sites, t.keys, t.coeff) for t in ts]
    return Terms{I}(out)
end

# Arithmetic
# ----------
@noinline function _slotmismatch(a::LocalOperator, b::LocalOperator)
    return throw(
        ArgumentError(
            "cannot combine a LocalOperator on $(nsites(a)) slots with one on $(nsites(b)); place " *
                "them first if they are meant to overlap"
        )
    )
end

function Base.:+(a::LocalOperator{I}, b::LocalOperator{I}) where {I}
    a.nsites == b.nsites || _slotmismatch(a, b)
    return LocalOperator{I}(a.terms + b.terms, a.nsites)
end
Base.:-(a::LocalOperator{I}, b::LocalOperator{I}) where {I} = a + (-b)
VectorInterface.scale(B::LocalOperator{I}, α::Number) where {I} =
    LocalOperator{I}(scale(B.terms, α), B.nsites)
Base.:*(α::Number, B::LocalOperator) = scale(B, α)
Base.:*(B::LocalOperator, α::Number) = scale(B, α)
Base.:/(B::LocalOperator, α::Number) = scale(B, inv(α))
Base.:-(B::LocalOperator) = scale(B, -1)

"""
    zero(B::LocalOperator)

The operator with no terms on the same `nsites(B)` slots. (There is no `zero(::Type)`: an empty
operator still has to know its `K`.)
"""
Base.zero(B::LocalOperator{I}) where {I} = LocalOperator{I}(Terms{I}(), B.nsites)

"""
    copy(B::LocalOperator) -> LocalOperator

A copy owning its own term vector, so `canonicalize!` on either side leaves the other alone; the
[`Term`](@ref)s themselves are shared, as for [`copy(::Terms)`](@ref).
"""
Base.copy(B::LocalOperator{I}) where {I} = LocalOperator{I}(copy(B.terms), B.nsites)

Base.:(==)(a::LocalOperator{I}, b::LocalOperator{I}) where {I} =
    a.nsites == b.nsites && a.terms == b.terms
Base.isapprox(a::LocalOperator{I}, b::LocalOperator{I}; kwargs...) where {I} =
    a.nsites == b.nsites && isapprox(a.terms, b.terms; kwargs...)

function Base.show(io::IO, B::LocalOperator{I}) where {I}
    canonicalize!(B.terms)
    print(io, "LocalOperator{", I, "}(nsites=", B.nsites, ": ")
    if isempty(B.terms.terms)
        print(io, "0")
    else
        join(io, ("$(t.coeff) * $(_slotbody(t))" for t in B.terms.terms), " + ")
    end
    return print(io, ")")
end
_slotbody(t::Term) = string("[ops=", ops(t), ", total=", total(t), "]")
