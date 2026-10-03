# The lattice-carrying container: a lattice, the finite-range terms placed on it, and any
# exponentially decaying channels. It is the one place the spaces and the operator meet, so it is
# where every letter is checked (when terms *enter*, so the error points at the line that added the
# term), and what `irrep_mpo`, `instantiate`, `islossless`, `jordan_mpo_tensors` and `H'` consume.
#
# It fills the four cells of finite/infinite × plain/decaying with one type — the shape `TermSum`
# lacked — and it holds `ExpSum` channels itself, which is what `MixedSum` used to be for. The term
# algebra underneath stays latticeless: `Terms` is still the bag `couple`, `dot` and placement
# return, and `mpo_terms` still returns one.

using Dictionaries: Dictionary, setwith!
using TensorKit: Sector

"""
    OperatorSum{I<:Sector, L<:AbstractLattice}

A Hamiltonian on a lattice: the lattice, a [`Terms`](@ref) bag of finite-range terms, and an
[`ExpSum`](@ref) of exponentially decaying channels (see [`expterm`](@ref)). This is what
[`irrep_mpo`](@ref), [`instantiate`](@ref), [`islossless`](@ref) and [`jordan_mpo_tensors`](@ref)
take, for a [`FiniteChain`](@ref) and an [`InfiniteChain`](@ref), with or without channels:

```julia
V = SU2Space(1 // 2 => 1); S = spin(V); b = dot(S, S)

H = opsum(FiniteChain(V, 8), (b[i] for i in 1:7))                         # finite
H = opsum(InfiniteChain(V), b[1])                                         # infinite
H = opsum(FiniteChain(V, 8), b[1], expterm(b[1]; decay = 0.4))            # finite, decaying
H = opsum(InfiniteChain(V), b[1], expterm(b[1]; decay = 0.4))             # infinite, decaying
mpo = irrep_mpo(H)
```

`OperatorSum(lat)` is the empty operator on `lat`, which may be a `FiniteChain`, an
`InfiniteChain`, or any iterable of one space per site (a `FiniteChain`). Fill it with
[`opsum!`](@ref) — one pass, in place, linear — or build it in one go with `opsum(lat, terms...)`.
`H + x` and `H += x` also work but *copy* `H`, so folding them over `M` terms is quadratic.

Terms are checked against the lattice as they are inserted: the symmetry sector, that every letter
exists on the space of the site it acts on, and — on a finite chain — that the site is in range. On
an infinite chain `H` is a *generating set* (`Σ_n translate(H, n·L)`), so every term and channel
must also be charge-neutral and no `K = 0` identity is allowed; that each translation class appears
once can only be seen across the whole set, so it is checked when the MPO is formed.

Like [`Terms`](@ref), it canonicalises lazily: `length`, iteration, `≈`, `==` and `show` report the
canonical operator. `length` and iteration run over the finite-range terms; the channels are
`H.channels`, and `isempty(H)` is true only when there are neither.

`H'` is the hermitian conjugate (see `adjoint`), `+`, `-`, `*` and `/` combine it with other
operators on the same lattice.
"""
struct OperatorSum{I <: Sector, L <: AbstractLattice}
    lattice::L
    terms::Terms{I}
    channels::ExpSum{I}
end

function OperatorSum(lat::AbstractLattice)
    I = _lattice_sectortype(lat)
    return OperatorSum{I, typeof(lat)}(lat, Terms{I}(), ExpSum{I}())
end
OperatorSum(sites) = OperatorSum(_tolattice(sites))

TensorKit.sectortype(::Type{<:OperatorSum{I}}) where {I} = I

Base.length(H::OperatorSum) = length(H.terms)
Base.isempty(H::OperatorSum) = isempty(H.terms) && isempty(H.channels)
Base.iterate(H::OperatorSum, args...) = iterate(H.terms, args...)
Base.eltype(::Type{<:OperatorSum{I}}) where {I} = Term{I}

"""
    canonicalize!(H::OperatorSum) -> H

Put the finite-range terms of `H` in normal form in place (see [`canonicalize!`](@ref) for `Terms`).
Channels are summed on insertion and need no normal form.
"""
canonicalize!(H::OperatorSum) = (canonicalize!(H.terms); H)

function Base.copy(H::OperatorSum{I, L}) where {I, L}
    return OperatorSum{I, L}(
        H.lattice, copy(H.terms), ExpSum{I}(copy(H.channels.channels))
    )
end
Base.zero(H::OperatorSum) = OperatorSum(H.lattice)
Base.empty(H::OperatorSum) = OperatorSum(H.lattice)

# --- inserting terms ------------------------------------------------------------------------------

"""
    opsum!(H::OperatorSum, terms...) -> H

Add terms to `H` **in place**, in one pass, so filling it with `M` terms costs `Θ(M)`:

```julia
H = OperatorSum(FiniteChain(V, N))
opsum!(H, (b[i] for i in 1:(N - 1)))
```

Each argument may be a [`Term`](@ref), a [`Terms`](@ref) bag, an [`ExpSum`](@ref) (from
[`expterm`](@ref)), or any iterable of those, nested arbitrarily. Everything is checked against the
lattice of `H` before anything is added, so a failing call leaves `H` untouched.

This is the linear route; `H + x` copies `H`.
"""
function opsum!(H::OperatorSum{I}, args...) where {I}
    terms = Term{I}[]
    chans = Dictionary{ExpKey{I}, ComplexF64}()
    for a in args
        _collect_op!(terms, chans, a, I)
    end
    _checkinsert(H.lattice, terms, chans)
    append!(H.terms.terms, terms)
    if !isempty(chans)
        for (k, v) in pairs(chans)
            setwith!(+, H.channels.channels, k, v)
        end
        filter!(!iszero, H.channels.channels)
    end
    return H
end

"""
    opsum(lat, terms...) -> OperatorSum

Accumulate `terms` into a new [`OperatorSum`](@ref) on `lat` (a [`FiniteChain`](@ref), an
[`InfiniteChain`](@ref) or a vector of spaces), in one pass. Same argument forms as
[`opsum!`](@ref), and checked against `lat` as they enter.
"""
opsum(lat::AbstractLattice, args...) = opsum!(OperatorSum(lat), args...)
opsum(sites::AbstractVector{<:ElementarySpace}, args...) = opsum!(OperatorSum(sites), args...)

_collect_op!(out, ch, t::Term{I}, ::Type{I}) where {I} = push!(out, t)
_collect_op!(out, ch, ts::Terms{I}, ::Type{I}) where {I} = append!(out, ts.terms)
function _collect_op!(out, ch, es::ExpSum{I}, ::Type{I}) where {I}
    for (k, v) in pairs(es.channels)
        setwith!(+, ch, k, v)
    end
    return ch
end
# a single non-`Term` operand that is not a bag: say what to do rather than recursing into it
_collect_op!(out, ch, a::Union{SiteOperator, LocalOperator}, ::Type) = throw(
    ArgumentError(
        "cannot add an unplaced $(nameof(typeof(a))) to an OperatorSum: place it on sites first " *
            "(`A[i]` for a SiteOperator, `B[i]` or `B[i, j]` for a LocalOperator)"
    )
)
_collect_op!(out, ch, ::OperatorSum, ::Type) = throw(
    ArgumentError("cannot add an OperatorSum to another with opsum!; combine them with `+`")
)
# a number is iterable, so it would recurse forever
_collect_op!(out, ch, a::Number, ::Type) = throw(
    ArgumentError("cannot add the number $a to an OperatorSum; scale an operator instead")
)
function _collect_op!(out, ch, itr, ::Type{I}) where {I}
    applicable(iterate, itr) || throw(
        ArgumentError(
            "cannot add a $(typeof(itr)) to an OperatorSum: expected a Term, Terms, ExpSum or an " *
                "iterable of those"
        )
    )
    for a in itr
        _collect_op!(out, ch, a, I)
    end
    return out
end
for T in (:Term, :Terms, :ExpSum)
    @eval _collect_op!(out, ch, ::$T{J}, ::Type{I}) where {I, J} = _wrongsector_op(I, J)
end
_wrongsector_op(I, J) = throw(
    ArgumentError("cannot add an operator over $J to an OperatorSum on a lattice of sector type $I")
)

# Everything that must hold of a term (or channel) the moment it enters, so the error names the line
# that added it: letters against spaces and site range (`_checklattice`), and on an infinite chain
# the generating-set rules that need no global view.
function _checkinsert(lat::AbstractLattice, terms::Vector{Term{I}}, chans) where {I}
    _checklattice(terms, lat)
    _checkgenerating(lat, terms, chans)
    return nothing
end

_checkgenerating(::FiniteChain, terms, chans) = nothing
function _checkgenerating(lat::InfiniteChain, terms::Vector{Term{I}}, chans) where {I}
    for t in terms
        _checkgenerator(t)
    end
    for k in keys(chans)
        _checkgenerator(k)
        # The representative is translated by multiples of `L`, which preserves the space, so its
        # letters can be checked here; on a finite chain every translate has a different site.
        _checklattice([k.term], lat)
    end
    return nothing
end

# --- arithmetic -----------------------------------------------------------------------------------

const _Summand{I} = Union{Term{I}, Terms{I}, ExpSum{I}}

Base.:+(H::OperatorSum, x::_Summand) = opsum!(copy(H), x)
Base.:+(x::_Summand, H::OperatorSum) = opsum!(copy(H), x)
function Base.:+(a::OperatorSum, b::OperatorSum)
    _samelattice(a, b, "+")
    return opsum!(copy(a), b.terms, b.channels)
end

VectorInterface.scale(H::OperatorSum{I, L}, α::Number) where {I, L} =
    OperatorSum{I, L}(H.lattice, scale(H.terms, α), scale(H.channels, α))
Base.:*(α::Number, H::OperatorSum) = scale(H, α)
Base.:*(H::OperatorSum, α::Number) = scale(H, α)
Base.:/(H::OperatorSum, α::Number) = scale(H, inv(α))
Base.:-(H::OperatorSum) = scale(H, -1)
Base.:-(a::OperatorSum, b::OperatorSum) = a + (-b)
Base.:-(H::OperatorSum, x::_Summand) = H + (-x)
Base.:-(x::_Summand, H::OperatorSum) = x + (-H)

function _samelattice(a::OperatorSum, b::OperatorSum, op)
    a.lattice == b.lattice || throw(
        ArgumentError(
            "cannot $op two OperatorSums on different lattices: $(a.lattice) and $(b.lattice)"
        )
    )
    return nothing
end

# `Terms + ExpSum` used to build a `MixedSum`; the bag and the channels now meet in a container.
for (A, B) in ((:Terms, :ExpSum), (:ExpSum, :Terms))
    @eval Base.:+(::$A, ::$B) = throw(
        ArgumentError(
            "a Terms bag and an ExpSum no longer combine on their own (MixedSum is gone): they meet " *
                "in an OperatorSum, `opsum(lat, h, expterm(h; decay = λ))`"
        )
    )
end

# --- comparison, display --------------------------------------------------------------------------

function _channelsapprox(a::ExpSum, b::ExpSum; kwargs...)
    length(a.channels) == length(b.channels) || return false
    for (k, v) in pairs(a.channels)
        haskey(b.channels, k) || return false
        isapprox(v, b.channels[k]; kwargs...) || return false
    end
    return true
end

"""
    isapprox(a::OperatorSum, b::OperatorSum; kwargs...)
    a ≈ b

Whether two operators live on the same lattice and carry the same terms and channels: the canonical
term sets and channel keys must be **equal**, the coefficients `≈`.
"""
Base.isapprox(a::OperatorSum{I}, b::OperatorSum{I}; kwargs...) where {I} =
    a.lattice == b.lattice && isapprox(a.terms, b.terms; kwargs...) &&
    _channelsapprox(a.channels, b.channels; kwargs...)
Base.:(==)(a::OperatorSum{I}, b::OperatorSum{I}) where {I} =
    a.lattice == b.lattice && a.terms == b.terms &&
    a.channels.channels == b.channels.channels

function _latticesummary(lat::AbstractLattice)
    kind = lat isa FiniteChain ? "FiniteChain" : "InfiniteChain"
    isempty(lat) && return string(kind, "([])")
    all(==(first(lat.spaces)), lat.spaces) &&
        return string(kind, "(", first(lat.spaces), ", ", length(lat), ")")
    return string(lat)
end

function Base.show(io::IO, H::OperatorSum)
    print(io, "OperatorSum(", _latticesummary(H.lattice), ", ", H.terms)
    isempty(H.channels) || print(io, ", ", H.channels)
    return print(io, ")")
end

# --- the operations that need the spaces ----------------------------------------------------------

"""
    adjoint(H::OperatorSum) -> OperatorSum
    H'

The hermitian conjugate, so that `H + H'` is hermitian:

```julia
F = fermion_ops()
T = opsum(lat, -t * couple(F.cd[i], F.c[i + 1]) for i in 1:(N - 1))
H = T + T'                                  # ≡ -t Σᵢ (c†ᵢcᵢ₊₁ + c†ᵢ₊₁cᵢ)
```

The adjoint of an alphabet letter is generally a *combination* of the dual charge's letters, which
only the physical space determines — which is why this lives on the container and a bare
[`Terms`](@ref) has no `h'`. Only the spaces of the sites a term touches are read.

Every term must have total charge `unit(I)`, the case a Hamiltonian term is in: a charged term's
adjoint carries the dual charge, so it is not a term of the same operator. An operator holding
exponentially decaying channels has no adjoint yet (`ArgumentError`).

Each distinct term *shape* costs one projection, memoised, so conjugating a whole Hamiltonian is
`O(number of distinct shapes)` rather than `O(number of terms)`.
"""
function Base.adjoint(H::OperatorSum{I, L}) where {I, L}
    isempty(H.channels) || throw(
        ArgumentError(
            "adjoint: this OperatorSum holds exponentially decaying channels, whose adjoint is " *
                "not supported yet. Conjugate the finite-range part and add the channels afterwards."
        )
    )
    return OperatorSum{I, L}(H.lattice, _adjoint_terms(H.terms, H.lattice), ExpSum{I}())
end

"""
    instantiate(H::OperatorSum) -> AbstractTensorMap

Materialize the operator on its lattice into a TensorKit `TensorMap`, summing each term (the dense
oracle). Supports identity (K=0), single-site field (K=1), and left-nested (caterpillar) coupling of
any K ≥ 2 sites. On a finite chain with exponentially decaying channels this is the geometric sum
truncated to the chain ([`chain_terms`](@ref)); an infinite chain has no finite operator.
"""
function instantiate(H::OperatorSum{I, <:FiniteChain}) where {I}
    isempty(H.lattice) && throw(ArgumentError("cannot instantiate over an empty lattice"))
    isempty(H) && throw(ArgumentError("cannot instantiate an empty operator"))
    return _instantiate_terms(chain_terms(H), H.lattice.spaces)
end
instantiate(::OperatorSum{I, <:InfiniteChain}) where {I} = throw(
    ArgumentError(
        "cannot instantiate an operator on an infinite chain; instantiate a finite window of it " *
            "(`mpo_terms_window`, then `instantiate(opsum(sites, terms))`)"
    )
)

"""
    chain_terms(H::OperatorSum) -> Terms

The literal term sum `H` stands for on a *finite* chain: the finite-range terms as written, plus
every translate (period 1) of every exponentially decaying channel that fits. This is what
`irrep_mpo(H)` represents there.
"""
function chain_terms(H::OperatorSum{I, <:FiniteChain}) where {I}
    return H.terms + expand_channels(H.channels, 1, length(H.lattice))
end

# `couple`/`dot` act on operators that have not met a lattice yet.
_opsumoperand(f) = throw(
    ArgumentError(
        "$f: an OperatorSum already lives on a lattice. Couple the unplaced or placed building " *
            "blocks (`SiteOperator`, `LocalOperator`, `Terms`) and add the result with `opsum!`."
    )
)
const _NotOperatorSum = Union{Term, Terms, SiteOperator, LocalOperator}
couple(::OperatorSum, args...; kwargs...) = _opsumoperand(:couple)
couple(::_NotOperatorSum, ::OperatorSum, args...; kwargs...) = _opsumoperand(:couple)
couple_channels(::OperatorSum, args...; kwargs...) = _opsumoperand(:couple_channels)
couple_channels(::_NotOperatorSum, ::OperatorSum, args...; kwargs...) =
    _opsumoperand(:couple_channels)
LinearAlgebra.dot(::OperatorSum, ::Any) = _opsumoperand(:dot)
LinearAlgebra.dot(::_NotOperatorSum, ::OperatorSum) = _opsumoperand(:dot)
LinearAlgebra.dot(::OperatorSum, ::OperatorSum) = _opsumoperand(:dot)
