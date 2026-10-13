# Stock bases and operator builders: one memoised basis per TensorKitTensors site and symmetry, and a
# builder projecting every neutral operator of that site into it. A basis is defined from an ordered
# list of TensorKitTensors operators (a two-site one contributes its factorised pieces), skipping what
# the symmetry does not admit or what is already spanned, then completed with matrix units; the order
# makes the common operators atomic.

import TensorKitTensors
using TensorKitTensors: SpinOperators, FermionOperators, HubbardOperators, TJOperators, BosonOperators

# The defining operators, in order.
const _SPIN_DEFINING = (:S_plus_S_min, :S_min_S_plus, :S_z, :S_z_S_z, :S_y_S_y, :S_exchange, :S_x)
const _FERMION_DEFINING = (:f_plus_f_min, :f_min_f_plus, :f_num)
const _BOSON_DEFINING = (:b_plus_b_min, :b_min_b_plus, :b_num)
const _HUBBARD_DEFINING = (
    :e_plus_e_min, :e_min_e_plus, :e_num, :ud_num, :S_plus_S_min, :S_min_S_plus, :S_exchange, :S_z,
    :e_hopping,   # its pieces are new only under an SU(2) particle symmetry, which has no `e⁺e⁻`
)

# What the builders hand out: the neutral one- and two-site operators of each module.
const _SPIN_OPS = (
    :S_x, :S_y, :S_z, :S_plus, :S_min,
    :S_x_S_x, :S_y_S_y, :S_z_S_z, :S_plus_S_min, :S_min_S_plus, :S_exchange,
)
const _FERMION_OPS = (:f_num, :f_plus_f_min, :f_min_f_plus, :f_plus_f_plus, :f_min_f_min, :f_hopping)
const _BOSON_OPS = (
    :b_plus, :b_min, :b_num, :b_plus_b_plus, :b_plus_b_min, :b_min_b_plus, :b_min_b_min, :b_hopping,
)
const _HUBBARD_OPS = (
    :e_num, :u_num, :d_num, :ud_num, :half_ud_num, :h_num,
    :S_x, :S_y, :S_z, :S_plus, :S_min,
    :u_plus_u_min, :d_plus_d_min, :u_min_u_plus, :d_min_d_plus,
    :u_min_d_min, :d_min_u_min, :u_plus_d_plus, :d_plus_u_plus,
    :u_min_u_min, :d_min_d_min, :u_plus_u_plus, :d_plus_d_plus,
    :e_plus_e_min, :e_min_e_plus, :e_hopping,
    :singlet_plus, :singlet_min,
    :S_plus_S_min, :S_min_S_plus, :S_exchange,
)

const _STOCK_CACHE = Dict{Any, Any}()

# `M.name(ComplexF64, args...; kwargs...)`, or `nothing` when the symmetry does not admit it
# (TensorKitTensors throws), it is charged, or it is zero.
function _tkt_operator(M::Module, name::Symbol, args, kwargs)
    isdefined(M, name) || return nothing
    t = try
        getfield(M, name)(ComplexF64, args...; kwargs...)
    catch err
        err isa Union{ArgumentError, MethodError, DomainError} || rethrow()
        return nothing
    end
    (numout(t) in (1, 2) && space(t) == (codomain(t) ← codomain(t))) || return nothing
    iszero(norm(t)) && return nothing
    return t
end

# The basis defined by `names` on `spacefn(args...; kwargs...)`; `args`/`kwargs` are also those of every
# TensorKitTensors call.
function _stock_basis(kind::Symbol, M::Module, spacefn, names, args, kwargs)
    return _cached(_STOCK_CACHE, (kind, args..., values(kwargs)...)) do
        V = spacefn(args...; kwargs...)
        elems = AbstractTensorMap[_identityelement(V)]
        for name in names
            t = _tkt_operator(M, name, args, kwargs)
            t === nothing && continue
            codomain(t)[1] == V || _invariant("TensorKitTensors operator $name does not act on $V")
            numout(t) == 1 ? push!(elems, insertrightunit(t)) : append!(elems, _factorise_pair(t))
        end
        _basis_from_elements(V, _completed_elements(V, elems); skipdependent = true)
    end
end

# Every operator of `names` the symmetry admits, projected into `basisfn(args...; kwargs...)`.
function _stock_ops(kind::Symbol, M::Module, basisfn, names, args, kwargs)
    return _cached(_STOCK_CACHE, (kind, args..., values(kwargs)...)) do
        B = basisfn(args...; kwargs...)
        ts = [name => _tkt_operator(M, name, args, kwargs) for name in names]
        filter!(p -> p.second !== nothing, ts)
        NamedTuple{Tuple(first.(ts))}(Tuple(project(last(p), B) for p in ts))
    end
end

"""
    spin_basis(symmetry = Trivial; spin = 1//2) -> Basis

The operator basis of TensorKitTensors' `spin_space(symmetry; spin)`: the identity, then `S⁺, S⁻` and
`Sᶻ` (no symmetry, U(1)), or `S⃗` (SU(2)), or the spin-flip-graded pieces (ℤ₂), completed to span every
operator on the space (the higher multipoles for `spin > 1/2`). Memoised.

See [`spin_ops`](@ref) for the operators expressed in it.
"""
function spin_basis(symmetry::Type{<:Sector} = Trivial; spin = 1 // 2)
    return _stock_basis(
        :spin_basis, SpinOperators, SpinOperators.spin_space, _SPIN_DEFINING, (symmetry,), (; spin)
    )
end

"""
    fermion_basis(symmetry = Trivial) -> Basis

The operator basis of TensorKitTensors' `fermion_space(symmetry)`: `𝟙, n, c†, c`. Memoised.
"""
function fermion_basis(symmetry::Type{<:Sector} = Trivial)
    return _stock_basis(
        :fermion_basis, FermionOperators, FermionOperators.fermion_space, _FERMION_DEFINING, (symmetry,), (;)
    )
end

"""
    boson_basis(symmetry = Trivial; cutoff) -> Basis

The operator basis of TensorKitTensors' `boson_space(symmetry; cutoff)`: the identity, `n`, `b†`, `b`,
completed to span every operator on the truncated space. Memoised.
"""
function boson_basis(symmetry::Type{<:Sector} = Trivial; cutoff::Integer)
    return _stock_basis(
        :boson_basis, BosonOperators, BosonOperators.boson_space, _BOSON_DEFINING, (symmetry,), (; cutoff)
    )
end

"""
    hubbard_basis(particle_symmetry = Trivial, spin_symmetry = Trivial) -> Basis

The operator basis of TensorKitTensors' `hubbard_space(particle_symmetry, spin_symmetry)`: the
identity, `n`, `nꜛꜜ`, the creation and annihilation operators `c†σ, cσ` (one spinor element each under
SU(2)), the spin operators, then completed to span every operator on the site (e.g. `n·c†` and the
on-site pair `Δ†, Δ`). Memoised.
"""
function hubbard_basis(particle_symmetry::Type{<:Sector} = Trivial, spin_symmetry::Type{<:Sector} = Trivial)
    return _stock_basis(
        :hubbard_basis, HubbardOperators, HubbardOperators.hubbard_space, _HUBBARD_DEFINING,
        (particle_symmetry, spin_symmetry), (;)
    )
end

"""
    tj_basis(particle_symmetry = Trivial, spin_symmetry = Trivial; slave_fermion = false) -> Basis

The operator basis of TensorKitTensors' `tj_space`: the t-J counterparts of [`hubbard_basis`](@ref)'s
defining operators, completed to span every operator on the (doubly-occupation-free) site. Memoised.
"""
function tj_basis(
        particle_symmetry::Type{<:Sector} = Trivial, spin_symmetry::Type{<:Sector} = Trivial;
        slave_fermion::Bool = false
    )
    return _stock_basis(
        :tj_basis, TJOperators, TJOperators.tj_space, _HUBBARD_DEFINING,
        (particle_symmetry, spin_symmetry), (; slave_fermion)
    )
end

"""
    spin_ops(symmetry = Trivial; spin = 1//2) -> NamedTuple
    fermion_ops(symmetry = Trivial) -> NamedTuple
    boson_ops(symmetry = Trivial; cutoff) -> NamedTuple
    hubbard_ops(particle_symmetry = Trivial, spin_symmetry = Trivial) -> NamedTuple
    tj_ops(particle_symmetry = Trivial, spin_symmetry = Trivial; slave_fermion = false) -> NamedTuple

Every charge-neutral operator TensorKitTensors defines on that site for that symmetry, projected into
the matching stock basis ([`spin_basis`](@ref), …) and named after its TensorKitTensors function: a
single-site operator is a `SiteOperator` (`ops.S_z[i]`), a two-site one an unplaced `LocalOperator`
(`ops.S_exchange[i, j]`). Operators the symmetry does not admit as neutral ones (`S_z` under SU(2),
`S_plus` under U(1)) are absent. Memoised.

```julia
ops = spin_ops(SU2Irrep)
H = opsum(FiniteChain(spin_basis(SU2Irrep), N), (ops.S_exchange[i, i + 1] for i in 1:(N - 1)))
```
"""
function spin_ops(symmetry::Type{<:Sector} = Trivial; spin = 1 // 2)
    return _stock_ops(:spin_ops, SpinOperators, spin_basis, _SPIN_OPS, (symmetry,), (; spin))
end
"See [`spin_ops`](@ref)."
function fermion_ops(symmetry::Type{<:Sector} = Trivial)
    return _stock_ops(:fermion_ops, FermionOperators, fermion_basis, _FERMION_OPS, (symmetry,), (;))
end
"See [`spin_ops`](@ref)."
function boson_ops(symmetry::Type{<:Sector} = Trivial; cutoff::Integer)
    return _stock_ops(:boson_ops, BosonOperators, boson_basis, _BOSON_OPS, (symmetry,), (; cutoff))
end
"See [`spin_ops`](@ref)."
function hubbard_ops(particle_symmetry::Type{<:Sector} = Trivial, spin_symmetry::Type{<:Sector} = Trivial)
    return _stock_ops(
        :hubbard_ops, HubbardOperators, hubbard_basis, _HUBBARD_OPS, (particle_symmetry, spin_symmetry), (;)
    )
end
"See [`spin_ops`](@ref)."
function tj_ops(
        particle_symmetry::Type{<:Sector} = Trivial, spin_symmetry::Type{<:Sector} = Trivial;
        slave_fermion::Bool = false
    )
    return _stock_ops(
        :tj_ops, TJOperators, tj_basis, _HUBBARD_OPS, (particle_symmetry, spin_symmetry), (; slave_fermion)
    )
end
