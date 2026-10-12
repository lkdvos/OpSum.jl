# Jordan-form MPO emission: one `SparseBlockTensorMap` per site, virtual legs `SumSpace`s with one level per
# bond index, reordered (start channel, rest, finish channel). Both identity channels are padded in at every
# internal bond even where the cover spent no index on them (at most `+2` per bond; padded channels are pure
# identity chains). `_force_finish!` (irrepgraph.jl) is the one thing that touches the cover for this path.

using BlockTensorKit: BlockTensorKit, SparseBlockTensorMap, SumSpace, eachspace
using TensorKit: BraidingTensor, Vect, unit, tensormaptype
using .IrrepTensorOperators: IrrepOperator

"""
    jordan_mpo_tensors(H::OperatorSum[, alg]) -> Vector{<:SparseBlockTensorMap}

`jordan_mpo_tensors(mpo(H, alg))`: compress `H` (see [`mpo`](@ref)) and emit it in **Jordan form**, one
`BlockTensorKit.SparseBlockTensorMap` per site (of the chain, or of the unit cell),
`W_i : B_{i-1} ⊗ V_i ← V_i ⊗ B_i`, whose virtual legs are `SumSpace`s with one level `Vect[I](charge => 1)`
per bond index, ordered `(start channel, everything else, finish channel)` with the identity at `(1, 1)`
and `(end, end)`. On a finite chain the boundary bonds are one-dimensional and trivially charged, so `H`
must have total charge `unit(I)`.

Diagonal unit-coefficient pass-throughs are emitted as `TensorKit.BraidingTensor`s, so a consumer can store
them as scalars. This is the shape MPSKit's `JordanMPOTensor` consumes (`instantiate(mpo(H))`); use
[`irrep_mpo_tensors`](@ref) for dense per-site `TensorMap`s.

An [`SVDBondAlgorithm`](@ref) bond basis mixes prefix states, so neither identity channel is a basis
vector and both are padded at every bond; a truncation that empties a bond is rejected, since the
identity corners need an index. Exponentially decaying channels on a finite chain have no Jordan form yet.
"""
jordan_mpo_tensors(H::OperatorSum, args...) = jordan_mpo_tensors(mpo(H, args...))

# Reorder one internal bond; `s`/`f` are the start/finish indices (`0` if the cover spent none). Slots
# `1`/`end` are reserved for them, live or not. Returns `(pos, sec)`: `pos[old] -> new`, `sec[new]` the charge.
function _jordan_bond(sec::Vector{I}, s::Int, f::Int) where {I}
    n = length(sec)
    iszero(n) && throw(
        ArgumentError(
            "a bond of the reduced MPO is zero-dimensional, so it cannot carry the Jordan identity " *
                "corners; this only happens under a truncating `SVDBondAlgorithm`"
        )
    )
    (iszero(s) || sec[s] == unit(I)) ||
        _invariant("the start channel carries the non-trivial charge $(sec[s])")
    (iszero(f) || sec[f] == unit(I)) ||
        _invariant("the finish channel carries the non-trivial charge $(sec[f])")
    (iszero(s) || s != f) || _invariant("the start and finish channels are the same bond index")

    pos = Vector{Int}(undef, n)
    newsec = I[unit(I)]                       # slot 1 — the start channel, live or padded
    iszero(s) || (pos[s] = 1)
    for j in 1:n
        (j == s || j == f) && continue
        push!(newsec, sec[j])
        pos[j] = length(newsec)
    end
    push!(newsec, unit(I))                    # slot end — the finish channel, live or padded
    iszero(f) || (pos[f] = length(newsec))
    return pos, newsec
end

"""
    jordan_mpo_tensors(Ws, bondsectors, starts, finishes, sites) -> Vector{<:SparseBlockTensorMap}

The reduced-data form of [`jordan_mpo_tensors`](@ref): `(Ws, bondsectors)` as an [`mpo`](@ref) holds them,
plus the per-bond start/finish indices from `_irrep_channels` (`0` where that channel is padded).

Two invariants are checked per site because a violation would silently drop operators: nothing may enter
the start column except from the start row, and the finish row may emit into nothing but the finish column.
"""
function jordan_mpo_tensors(
        Ws::Vector{<:SparseMatrixCSC{SiteOperator{I}}}, bondsectors::Vector{Vector{I}},
        starts::Vector{Int}, finishes::Vector{Int}, sites
    ) where {I}
    N = length(Ws)
    N == length(sites) ||
        throw(ArgumentError("got $N bond matrices for $(length(sites)) sites"))
    bondsectors[N] == I[unit(I)] || throw(
        ArgumentError(
            "the right boundary bond is $(bondsectors[N]), not the trivial `[$(unit(I))]`: only a " *
                "total-charge-`unit` operator has a Jordan MPO"
        )
    )

    # `pos[i]` / `jsec[i]` describe the bond to the *left* of site i; the boundaries stay one-dimensional.
    pos = Vector{Vector{Int}}(undef, N + 1)
    jsec = Vector{Vector{I}}(undef, N + 1)
    pos[1], jsec[1] = Int[1], I[unit(I)]
    pos[N + 1], jsec[N + 1] = Int[1], I[unit(I)]
    for i in 1:(N - 1)
        pos[i + 1], jsec[i + 1] = _jordan_bond(bondsectors[i], starts[i], finishes[i])
    end
    return _jordan_tensors(Ws, pos, jsec, sites)
end

# The emission proper, shared by the finite chain and the infinite unit cell: `pos[i]` maps the old indices
# of the bond left of site `i` to Jordan slots, `jsec[i]` gives those slots' charges (`length == N + 1`).
function _jordan_tensors(
        Ws::Vector{<:SparseMatrixCSC{SiteOperator{I}}}, pos::Vector{Vector{Int}},
        jsec::Vector{Vector{I}}, sites
    ) where {I}
    N = length(Ws)
    S = typeof(_space(first(sites)))
    TT = AbstractTensorMap{ComplexF64, S, 2, 2}     # abstract: `BraidingTensor`s stay braidings
    spaces = map(sec -> SumSpace([Vect[I](c => 1) for c in sec]), jsec)

    return map(1:N) do i
        site = sites[i]
        V = _space(site)
        secL, secR = jsec[i], jsec[i + 1]
        nrows, ncols = length(secL), length(secR)

        # 1. permute the reduced entries into Jordan order (a bijection)
        entries = Dictionary{Tuple{Int, Int}, SiteOperator{I}}()
        for (idx, localop) in storedpairs(Ws[i])
            l, r = Tuple(idx)
            op = prune(localop)
            isempty(op) && continue
            insert!(entries, (pos[i][l], pos[i + 1][r]), op)
        end

        # 2. the structural invariants, then the identity corners (padded where missing)
        for (row, col) in keys(entries)
            (ncols > 1 && col == 1 && row != 1) &&
                _invariant("row $row of site $i enters the Jordan start channel")
            (nrows > 1 && row == nrows && col != ncols) &&
                _invariant("the Jordan finish channel of site $i emits into column $col")
        end
        for corner in ((1, 1, ncols > 1), (nrows, ncols, nrows > 1))
            row, col, needed = corner
            needed || continue
            op = get(entries, (row, col), nothing)
            if op === nothing
                insert!(entries, (row, col), one(SiteOperator{I}))
            elseif !isone(op)
                _invariant("the Jordan corner ($row, $col) of site $i is not the identity")
            end
        end

        # 3. emit; a unit-coefficient pass-through on a charge-preserving slot is a braiding
        W = SparseBlockTensorMap{TT}(undef, spaces[i] ⊗ V ← V ⊗ spaces[i + 1])
        blocks = Dictionary{Tuple{IrrepOperator{I}, I, I}, tensormaptype(S, 2, 2, ComplexF64)}()
        for ((row, col), op) in pairs(entries)
            bL, bR = secL[row], secR[col]
            if isone(op)
                bL == bR ||
                    _invariant("a pass-through bond entry connects charge $bL to charge $bR")
                W[row, 1, 1, col] = BraidingTensor{ComplexF64, S, Vector{ComplexF64}}(
                    eachspace(W)[row, 1, 1, col]
                )
                continue
            end
            blk = nothing
            for (letter, coeff) in pairs(op)
                B = get!(() -> _letter_block(letter, site, bL, bR), blocks, (letter, bL, bR))
                blk = blk === nothing ? coeff * B : blk + coeff * B
            end
            W[row, 1, 1, col] = blk
        end
        return W
    end
end
