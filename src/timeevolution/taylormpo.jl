using SparseArrays: SparseMatrixCSC, nonzeros
using TensorKit: Sector, dim

"""
    TaylorEntry{I}

A polynomial in `τ` with [`SiteOperator`](@ref) coefficients: parallel vectors of strictly ascending `powers`
and the nonzero `ops` multiplying them. `+` adds, `scaled(e, c, k)` multiplies by `c τ^k`, `evaluate(e, τ)`
gives the [`SiteOperator`](@ref) at a value of `τ`.
"""
struct TaylorEntry{I <: Sector}
    powers::Vector{Int}
    ops::Vector{SiteOperator{I}}
end
TaylorEntry{I}() where {I} = TaylorEntry{I}(Int[], SiteOperator{I}[])
function TaylorEntry(op::SiteOperator{I}, power::Integer = 0) where {I}
    return isempty(op) ? TaylorEntry{I}() : TaylorEntry{I}([Int(power)], [op])
end

Base.zero(::Union{TaylorEntry{I}, Type{TaylorEntry{I}}}) where {I} = TaylorEntry{I}()
Base.iszero(e::TaylorEntry) = isempty(e.powers)      # an entry is always kept pruned
Base.length(e::TaylorEntry) = length(e.powers)
Base.:(==)(x::TaylorEntry{I}, y::TaylorEntry{I}) where {I} = x.powers == y.powers && x.ops == y.ops
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
        isempty(op) && continue
        push!(powers, p)
        push!(ops, op)
    end
    return TaylorEntry{I}(powers, ops)
end

# `c τ^k e`
function scaled(e::TaylorEntry{I}, c::Number, k::Integer = 0) where {I}
    iszero(c) && return TaylorEntry{I}()
    return TaylorEntry{I}(e.powers .+ Int(k), [op * c for op in e.ops])
end

"""
    evaluate(e::TaylorEntry, τ::Number) -> SiteOperator
    evaluate(W::TaylorMPO, τ::Number) -> Vector{SparseMatrixCSC{SiteOperator}}

The value at `τ`: of one polynomial entry, or of every entry of a [`TaylorMPO`](@ref) (its bond matrices, the
form [`mpo_tensors`](@ref) takes).
"""
function evaluate(e::TaylorEntry{I}, τ::Number) where {I}
    acc = zero(SiteOperator{I})
    for (p, op) in zip(e.powers, e.ops)
        acc += op * τ^p
    end
    return acc
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

"""
    TaylorMPO{I<:Sector, L<:AbstractLattice}

An MPO of the time-evolution operator `exp(τH)`, correct through `τ^order`, with the time left symbolic: the
bond matrices `Ws[i]` of site `i` hold polynomials in `τ` (`TaylorEntry`s), and `bondsectors[i]` is the charge
of each index of the bond right of site `i`, as in [`SymbolicMPO`](@ref) (on an [`InfiniteChain`](@ref) bond
`L` is the one left of site 1; the right boundary of a finite chain is the single index of charge `unit`).
Built by [`taylor_mpo`](@ref); [`evaluate`](@ref)`(W, τ)`, or `W(τ)`, gives the bond matrices at a value of `τ`
and `mpo_tensors(W, τ)` the symmetric site tensors.
"""
struct TaylorMPO{I <: Sector, L <: AbstractLattice}
    lattice::L
    Ws::Vector{SparseMatrixCSC{TaylorEntry{I}, Int}}
    bondsectors::Vector{Vector{I}}
    order::Int
end

Base.length(W::TaylorMPO) = length(W.Ws)
Base.getindex(W::TaylorMPO, i::Integer) = W.Ws[i]

function Base.show(io::IO, W::TaylorMPO{I, L}) where {I, L}
    D = [sum(dim, sec; init = 0) for sec in W.bondsectors]
    return print(io, "TaylorMPO{", I, "}(", nameof(L), ", order = ", W.order, ", N = ", length(W), ", D = ", D, ")")
end

function evaluate(W::TaylorMPO{I}, τ::Number) where {I}
    return map(W.Ws) do M
        vals = SiteOperator{I}[evaluate(e, τ) for e in nonzeros(M)]
        return SparseMatrixCSC(size(M, 1), size(M, 2), copy(M.colptr), copy(M.rowval), vals)
    end
end

(W::TaylorMPO)(τ::Number) = evaluate(W, τ)

mpo_tensors(W::TaylorMPO, τ::Number) = reduced_tensors(evaluate(W, τ), W.bondsectors, W.lattice)
