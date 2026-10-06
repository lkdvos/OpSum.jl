# Reduced MPO for the ITO automaton: `irrep_mpo` compresses an `OperatorSum` (a `Terms` bag bound to a
# lattice) into reduced bond matrices plus per-bond charge sectors, by running a per-bond-sector sweep
# (irrepgraph.jl, strategy chosen by `algorithms.jl`) over the flat `ITOTermTable`. `mpo_terms`
# inverts it at the reduced level (faithfulness) and `irrep_mpo_tensors` assembles the symmetric
# `TensorMap`s. Any arity K ≥ 0.
#
# The lattice enters through the `OperatorSum`, where the letters were checked against the spaces of
# the sites they act on when the terms went in.

using SparseArrays: SparseMatrixCSC
using TensorKit: Vect, ElementarySpace, fusiontrees, permute, dim, unit, isomorphism, @tensor,
    ncon, removeunit
using .IrrepTensorOperators: IrrepOperator

"""
    FiniteMPO{I<:Sector}

A reduced MPO for an open chain of `length(Ws)` sites, as [`irrep_mpo`](@ref) returns it.

`Ws[i]` is the reduced bond matrix of site `i` (`Ws[i] : bond i-1 → bond i`, entries are ITO letters
times reduced coefficients) and `bondsectors[i]` gives the bond charge of each bond index to the
*right* of site `i`, so `length(bondsectors[i])` is the reduced bond dimension there and
`sum(dim, bondsectors[i])` the dense-equivalent one. Bond `0` is the one-dimensional trivial-charge
left boundary.

It destructures as `Ws, bondsectors = irrep_mpo(H)`, exactly like an [`InfiniteMPO`](@ref).
"""
struct FiniteMPO{I <: Sector}
    Ws::Vector{SparseMatrixCSC{SiteOperator{I}, Int}}
    bondsectors::Vector{Vector{I}}
end

Base.length(H::FiniteMPO) = length(H.Ws)

# iterate/destructure as `(Ws, bondsectors)`, the same contract as `InfiniteMPO`
Base.iterate(H::FiniteMPO, state::Int = 1) =
    state == 1 ? (H.Ws, 2) : state == 2 ? (H.bondsectors, 3) : nothing

function Base.show(io::IO, H::FiniteMPO{I}) where {I}
    D = [sum(dim, sec; init = 0) for sec in H.bondsectors]
    return print(io, "FiniteMPO{", I, "}(N = ", length(H), ", D = ", D, ")")
end

"""
    irrep_mpo(H::OperatorSum[, alg]) -> FiniteMPO | InfiniteMPO

Compress the Hamiltonian `H` — an [`OperatorSum`](@ref), which carries its lattice — into a reduced
MPO. A [`FiniteChain`](@ref) gives a [`FiniteMPO`](@ref), an [`InfiniteChain`](@ref) an
[`InfiniteMPO`](@ref), with or without exponentially decaying channels (see [`expterm`](@ref)); the
signature is the same in all four cases:

```julia
irrep_mpo(opsum(FiniteChain(V, N), h))        # -> FiniteMPO
irrep_mpo(opsum(InfiniteChain(V), h))         # -> InfiniteMPO
```

On a `FiniteChain` the operator is `H` as written, on `N = length(lat)` sites. On an `InfiniteChain`
of `L` sites `H` is a **generating set**: the operator represented is `Σ_{n ∈ ℤ} translate(H, n·L)`,
so each translation class must appear exactly once (see [`unitcell_terms`](@ref)); that is checked
here, as it can only be seen across the whole set. The letters were already checked against the
spaces when the terms entered `H`.

A channel's period is fixed by the lattice: on an `InfiniteChain` of `L` sites its entry and exit step
by `L`, while on a `FiniteChain` every site is a possible entry, i.e. the operator represented is the
whole geometric sum truncated to the chain — [`chain_terms`](@ref)`(H)` spells it out. Each channel
becomes a single bond index with `λ` on its diagonal, so its cost is independent of the interaction
range.

`alg` is the algorithm selector: `BipartiteAlgorithm()` (the default, lossless minimum vertex cover)
or `SVDBondAlgorithm(trunc; sweep)`, whose `sweep` picks between the two truncation semantics
(`IndependentSVD`, the default, versus `SequentialSVD`). Each selector names a
[`BondStrategy`](@ref); the sweeps live in irrepgraph.jl. `SVDBondAlgorithm` is available on a
`FiniteChain` without channels only.
"""
function irrep_mpo(
        H::OperatorSum{I, <:FiniteChain},
        alg::Union{BipartiteAlgorithm, SVDBondAlgorithm} = BipartiteAlgorithm()
    ) where {I}
    N = length(H.lattice)
    if isempty(H.channels)
        tt = ITOTermTable(H.terms, N)
        return FiniteMPO(_irrep_sweep(tt, nvertices(tt), bondstrategy(alg))...)
    end
    alg isa SVDBondAlgorithm && throw(
        ArgumentError(
            "SVDBondAlgorithm cannot compress an exponentially decaying interaction: it needs the " *
                "whole bond coefficient matrix up front, and a geometric channel is a bond index " *
                "with a diagonal rather than a column of it. Use BipartiteAlgorithm() (the default)."
        )
    )
    return FiniteMPO(
        _irrep_sweep(
            ITOTermTable(H.terms, N), N, VertexCover();
            channels = _lower_channels(H.channels, 1)
        )...
    )
end

irrep_mpo(H::OperatorSum{I, <:InfiniteChain}, ::BipartiteAlgorithm = BipartiteAlgorithm()) where {I} =
    _infinite_window(unitcell_terms(H))

function irrep_mpo(::OperatorSum{I, <:InfiniteChain}, ::SVDBondAlgorithm) where {I}
    throw(
        ArgumentError(
            "SVDBondAlgorithm does not extend to infinite chains: it compresses each bond " *
                "independently against a vacuum-terminated layout, with no bond basis that closes " *
                "on itself. Use BipartiteAlgorithm() (the default)."
        )
    )
end

# The `(h, lat)` forms are gone: the lattice is bound where the terms are added.
irrep_mpo(::Union{Term, Terms, ExpSum}, args...) = throw(
    ArgumentError(
        "irrep_mpo no longer takes a lattice: put the terms on one first, " *
            "`irrep_mpo(opsum(lat, h))` (see `OperatorSum`)"
    )
)

"""
    mpo_terms(Ws, bondsectors; leftidx = 1, rightidx = nothing) -> Terms
    mpo_terms(H::FiniteMPO; kwargs...) -> Terms

Reconstruct the operator generated by a reduced MPO (inverse of `irrep_mpo` at the reduced level):
enumerate every path through the bond matrices, multiply the reduced coefficients along it, and read
the active ITO letters (skipping pass-through) together with their outgoing bond charges — which are
exactly the caterpillar running bonds a term carries.

No lattice is involved on either side: the bond data names charges but not physical spaces, and the
result is a latticeless [`Terms`](@ref) bag. For the lossless bipartite compression this recovers the
original operator exactly: `mpo_terms(irrep_mpo(H)) ≈ H.terms` is the faithfulness check, which
[`islossless`](@ref) packages up.

`leftidx` and `rightidx` are the boundary vectors. The finite defaults (`1`, and "accept any final
index", which is what a vacuum-terminated chain needs since its last bond is one-dimensional) become
the two identity channels for an infinite MPO tiled over a window — see `mpo_terms_window`.
"""
function mpo_terms(
        Ws::Vector{<:SparseMatrixCSC{SiteOperator{I}}},
        bondsectors::Vector{Vector{I}};
        leftidx::Int = 1, rightidx::Union{Nothing, Int} = nothing
    ) where {I}
    N = length(Ws)
    N == 0 && return Terms{I}()

    cols = Tuple{Vector{Int}, Vector{ITOKey{I}}, ComplexF64}[]
    function walk(i, leftidx, coeff, sites, keys)
        if i > N
            # `leftidx` here is the *running* bond index the path arrived on; a finite chain accepts
            # any final index, an infinite window only the done channel
            (rightidx === nothing || leftidx == rightidx) || return
            push!(cols, (copy(sites), copy(keys), coeff))
            return
        end
        for (idx, localop) in storedpairs(Ws[i])
            l, r = Tuple(idx)
            l == leftidx || continue
            for (letter, c) in pairs(localop)
                # the outgoing bond charge of an active (non-pass-through) letter is that position's
                # running caterpillar bond `bₖ`; idle sites contribute nothing
                if ispassthrough(letter)
                    walk(i + 1, r, coeff * c, sites, keys)
                else
                    push!(sites, i)
                    push!(keys, ITOKey{I}(letter, bondsectors[i][r], 1))
                    walk(i + 1, r, coeff * c, sites, keys)
                    pop!(sites)
                    pop!(keys)
                end
            end
        end
        return
    end
    walk(1, leftidx, ComplexF64(1), Int[], ITOKey{I}[])

    return Terms{I}(Term{I}[Term{I}(s, k, c) for (s, k, c) in cols])
end

mpo_terms(H::FiniteMPO; kwargs...) = mpo_terms(H.Ws, H.bondsectors; kwargs...)

# Site tensors `W_i : B_{i-1} ⊗ V_i ← V_i ⊗ B_i` (MPSKit leg convention), virtual legs built from the
# per-sector bond multiplicities.

# GradedSpace on a bond with the given per-index charges (multiplicity = count per sector).
function _bond_space(sec::Vector{I}) where {I}
    counts = Dict{I, Int}()
    for c in sec
        counts[c] = get(counts, c, 0) + 1
    end
    return Vect[I](counts)
end

# degeneracy index (within its charge sector) of each bond index
function _deg_indices(sec::Vector{I}) where {I}
    counts = Dict{I, Int}()
    out = Vector{Int}(undef, length(sec))
    for (j, c) in enumerate(sec)
        counts[c] = get(counts, c, 0) + 1
        out[j] = counts[c]
    end
    return out
end

# Contract one letter's accumulated bond coupler into the site tensor. Split out so that `@tensor`
# specialises on the concrete tensor types even though the per-site memo tables below are
# heterogeneously typed.
function _add_bond_entry!(W, O, κ)
    @tensor W[bl o; ii br] += O[o; ii cc] * κ[bl cc; br]
    return W
end

# Write one reduced coefficient into a coupler block. Also a function barrier: this runs once per
# *stored bond entry* — `O(D_L · D_R)` of them, against `O(d²)` contractions above — so it is the loop
# that must not stay dynamically dispatched off the memo tables' `Any` element type.
function _add_coupler_coeff!(κ, f1, f2, dL::Int, dR::Int, coeff)
    κ[f1, f2][dL, 1, dR] += coeff
    return κ
end

# The on-site tensor for one ITO letter: `V ← V ⊗ Vect[I](c => 1)`. Pass-through (c = unit) carries a
# trivial charge leg so the bond contraction below is uniform (`instantiate` would drop it, returning
# bare `id(V)`).
function _letter_optensor(letter::IrrepOperator{I}, V) where {I}
    Vc = Vect[I](letter.c => 1)
    return ispassthrough(letter) ? isomorphism(ComplexF64, V ← V ⊗ Vc) : instantiate(letter, V)
end

# One letter's site-tensor block for a single `(bL, bR)` bond-charge pair, coefficient `1`: the on-site
# tensor coupled `(bL, c) → bR` (running bond first, matching the caterpillar). Built from the same
# `_add_coupler_coeff!`/`_add_bond_entry!` primitives `_mpo_tensors` uses to batch many bond entries
# into one shared, bond-space-sized coupler before a single `@tensor`; here the coupler and result
# instead span only the single-index spaces `Vect[I](bL/bR => 1)`, since a caller with no bond
# degeneracy to batch over (i.e. `jordan_mpo_tensors`) wants one already-contracted block per
# `(letter, bL, bR)` triple.
function _letter_block(letter::IrrepOperator{I}, V, bL::I, bR::I) where {I}
    c = letter.c
    O = _letter_optensor(letter, V)
    f1 = only(fusiontrees((bL, c), bR, (false, false)))
    f2 = only(fusiontrees((bR,), bR, (false,)))
    κ = zeros(ComplexF64, Vect[I](bL => 1) ⊗ Vect[I](c => 1) ← Vect[I](bR => 1))
    _add_coupler_coeff!(κ, f1, f2, 1, 1, ComplexF64(1))
    blk = zeros(ComplexF64, Vect[I](bL => 1) ⊗ V ← V ⊗ Vect[I](bR => 1))
    _add_bond_entry!(blk, O, κ)
    return blk
end

"""
    irrep_mpo_tensors(H::FiniteMPO, lat) -> Vector{<:AbstractTensorMap}
    irrep_mpo_tensors(Ws, bondsectors, lat) -> Vector{<:AbstractTensorMap}

Assemble the symmetric MPO from the reduced bond matrices + bond sectors (from `irrep_mpo`). Site
tensor `W_i : B_{i-1} ⊗ V_i ← V_i ⊗ B_i` (MPSKit convention); the boundary bonds `B_0`, `B_N` are
one-dimensional.
Each reduced entry `(l, r, letter, coeff)` places the ITO letter's tensor into the `(l → r)` bond
transition weighted by `coeff`, coupling the operator charge into the bond via the (forward)
fusion `b_L ⊗ c → b_R`.
"""
function irrep_mpo_tensors(
        Ws::Vector{<:SparseMatrixCSC{SiteOperator{I}}},
        bondsectors::Vector{Vector{I}}, sites
    ) where {I}
    lat = _tolattice(sites)
    # Every internal bond is shared by two site tensors, so build its graded space and degeneracy
    # indices once rather than twice (as `Bright`/`degR` of site i and `Bleft`/`degL` of site i+1).
    # `bsecs[i]` is the bond to the *left* of site i; `bsecs[1]` is the trivial left boundary.
    N = length(Ws)
    bsecs = Vector{Vector{I}}(undef, N + 1)
    bsecs[1] = I[unit(I)]
    for i in 1:N
        bsecs[i + 1] = bondsectors[i]
    end
    return _mpo_tensors(Ws, bsecs, lat)
end

irrep_mpo_tensors(H::FiniteMPO, lat) = irrep_mpo_tensors(H.Ws, H.bondsectors, lat)

"""
    irrep_mpo_tensors(H::InfiniteMPO, lattice::InfiniteChain) -> Vector{<:AbstractTensorMap}

Assemble the `L` symmetric site tensors of an infinite MPO. Identical to the finite assembly except
that the bond to the left of site 1 is bond `L` rather than the vacuum, so the returned tensors tile:
`space(Ts[1], 1) == space(Ts[L], 4)'`.
"""
function irrep_mpo_tensors(H::InfiniteMPO{I}, lattice::InfiniteChain) where {I}
    L = length(H)
    length(lattice) == L ||
        throw(ArgumentError("unit cell length mismatch: MPO has $L sites, lattice has $(length(lattice))"))
    # the only change from the finite case: bond 0 *is* bond L (wrap-around instead of vacuum)
    bsecs = Vector{Vector{I}}(undef, L + 1)
    bsecs[1] = H.bondsectors[L]
    for j in 1:L
        bsecs[j + 1] = H.bondsectors[j]
    end
    return _mpo_tensors(H.Ws, bsecs, lattice)
end

# Shared assembly: `bsecs[i]` is the bond to the *left* of site i (length N+1), so the finite and
# infinite paths differ only in what `bsecs[1]` is.
function _mpo_tensors(
        Ws::Vector{<:SparseMatrixCSC{SiteOperator{I}}},
        bsecs::Vector{Vector{I}}, sites
    ) where {I}
    N = length(Ws)
    bspaces = map(_bond_space, bsecs)
    bdegs = map(_deg_indices, bsecs)

    # `map` infers a concrete `Vector{<:AbstractTensorMap}` element type (all site tensors share
    # the same `B ⊗ V ← V ⊗ B` space *type*), unlike an untyped preallocated buffer.
    return map(1:N) do i
        V = sites[i]
        secL, secR = bsecs[i], bsecs[i + 1]
        Bleft, Bright = bspaces[i], bspaces[i + 1]
        degL, degR = bdegs[i], bdegs[i + 1]

        W = zeros(ComplexF64, Bleft ⊗ V ← V ⊗ Bright)

        # Per *letter*, not per entry: κ_letter collects every reduced coefficient carrying that
        # letter, so it is `O(d²)` contractions rather than `O(D_L · D_R)`.
        ops = Dictionary{IrrepOperator{I}, Any}()
        couplers = Dictionary{IrrepOperator{I}, Any}()
        f1s = Dictionary{Tuple{I, I, I}, Any}()
        f2s = Dictionary{I, Any}()

        for (idx, localop) in storedpairs(Ws[i])
            l, r = Tuple(idx)
            bL, dL = secL[l], degL[l]
            bR, dR = secR[r], degR[r]
            for (letter, coeff) in pairs(localop)
                c = letter.c
                get!(() -> _letter_optensor(letter, V), ops, letter)
                # Fusion `(bL, c) → bR`: running bond FIRST, matching the caterpillar. Charge-first
                # flips the sign at antisymmetric inner vertices (1⊗1→1, so K ≥ 3).
                κ = get!(couplers, letter) do
                    return zeros(ComplexF64, Bleft ⊗ Vect[I](c => 1) ← Bright)
                end
                f1 = get!(() -> only(fusiontrees((bL, c), bR, (false, false))), f1s, (bL, c, bR))
                f2 = get!(() -> only(fusiontrees((bR,), bR, (false,))), f2s, bR)
                # `(l, r)` determines `(bL, dL, bR, dR)` and the letter fixes `c`, so distinct stored
                # entries always address distinct slots; the `+=` only matters if a `SiteOperator`
                # carries one letter twice, which `+` prevents.
                _add_coupler_coeff!(κ, f1, f2, dL, dR, coeff)
            end
        end

        for (letter, κ) in pairs(couplers)
            _add_bond_entry!(W, ops[letter], κ)
        end
        return W
    end
end

"""
    islossless(H::OperatorSum[, alg]) -> Bool

Whether the compressed MPO reproduces `H` exactly: reconstruct the operator the reduced MPO generates
with [`mpo_terms`](@ref) and compare it against `H` with `≈`, i.e. term *set* exactly and coefficients
approximately.

This is the primary correctness check for a construction — cheap, purely symbolic, independent of `N`,
and valid for fermionic sectors, where densifying is not a well-defined operation. `H` is compared as
[`chain_terms`](@ref)`(H)` (channels expanded). Finite chains only.

Only meaningful for lossless compression: after a truncating [`SVDBondAlgorithm`](@ref) `false` is the
expected answer rather than a bug.
"""
function islossless(H::OperatorSum{I, <:FiniteChain}, args...) where {I}
    return mpo_terms(irrep_mpo(H, args...)) ≈ chain_terms(H)
end

islossless(::OperatorSum{I, <:InfiniteChain}, args...) where {I} = throw(
    ArgumentError("islossless is not supported on an infinite chain")
)

"""
    mpo_tensormap(Ts::AbstractVector) -> AbstractTensorMap

Contract a chain of MPO site tensors — as [`irrep_mpo_tensors`](@ref) returns them, or
[`jordan_mpo_tensors`](@ref) after `map(TensorMap, ·)` — into the single `N`-site operator, in
[`instantiate`](@ref)'s leg convention: codomain `o₁…o_N`, domain `i₁…i_N` and the trailing
total-charge leg. The trivial left boundary bond is dropped.

The tensor-level oracle, `mpo_tensormap(irrep_mpo_tensors(irrep_mpo(H), H.lattice)) ≈
instantiate(H)`. Exponential in `N`, so small systems only — but it stays inside TensorKit and never
materialises a dense array, so unlike `convert(Array, ·)` it is valid for fermionic sectors.
"""
function mpo_tensormap(Ts::AbstractVector)
    N = length(Ts)
    N >= 1 || throw(ArgumentError("mpo_tensormap: need at least one site tensor"))
    net = [[i == 1 ? -(2N + 1) : i - 1, -i, -(N + i), i == N ? -(2N + 2) : i] for i in 1:N]
    O = ncon(Ts, net)                # legs: o₁…o_N, i₁…i_N, bL, bR
    O = removeunit(O, 2N + 1)        # drop the trivial left boundary
    return permute(O, (ntuple(identity, N), (ntuple(i -> N + i, N)..., 2N + 1)))
end
