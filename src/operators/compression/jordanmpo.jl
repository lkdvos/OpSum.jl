# Jordan-form emission: one `SparseBlockTensorMap` per site, bond indices reordered (start, rest, finish).
# Both identity channels are padded in at every internal bond where the cover spent no index on them.

using BlockTensorKit: BlockTensorKit, SparseBlockTensorMap, SumSpace, eachspace
using TensorKit: BraidingTensor, Vect, unit, tensormaptype
using .IrrepTensorOperators: IrrepOperator

"""
    jordan_mpo_tensors(H::OperatorSum[, alg]) -> Vector{<:SparseBlockTensorMap}

`jordan_mpo_tensors(mpo(H, alg))`: the compressed MPO in **Jordan form**, one `SparseBlockTensorMap` per
site (of the chain or the unit cell), `W_i : B_{i-1} ⊗ V_i ← V_i ⊗ B_i`, with bond indices ordered
`(start channel, everything else, finish channel)` and identities at `(1, 1)` and `(end, end)` emitted as
`BraidingTensor`s. On a finite chain `H` must have total charge `unit(I)`. This is the shape MPSKit's
`JordanMPOTensor` consumes; use [`irrep_mpo_tensors`](@ref) for dense `TensorMap`s. Not available for
decaying channels on a finite chain; a truncating [`SVDBondAlgorithm`](@ref) that empties a bond is rejected.
"""
jordan_mpo_tensors(H::OperatorSum, args...) = jordan_mpo_tensors(mpo(H, args...))

# Reorder one bond; `s`/`f` are the start/finish indices (`0` if absent); slots `1`/`end` are reserved for
# them. Returns `(pos, sec)`: `pos[old] -> new`, `sec[new]` the charge.
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

# Steps 1-2 of the emission, symbolic: the entries of site `i` in Jordan order. `posL`/`posR` map the old
# indices of its two bonds to Jordan slots, `nrows`/`ncols` are the Jordan bond dimensions.
function _jordan_entries(
        Wi::SparseMatrixCSC{SiteOperator{I}}, posL::Vector{Int}, posR::Vector{Int},
        nrows::Int, ncols::Int, i::Int
    ) where {I}
    # 1. permute the reduced entries into Jordan order (a bijection)
    entries = Dictionary{Tuple{Int, Int}, SiteOperator{I}}()
    for (idx, localop) in storedpairs(Wi)
        l, r = Tuple(idx)
        op = prune(localop)
        isempty(op) && continue
        insert!(entries, (posL[l], posR[r]), op)
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
    return entries
end

# `pos[i]` maps the old indices of the bond left of site `i` to Jordan slots, `jsec[i]` their charges.
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
        entries = _jordan_entries(Ws[i], pos[i], pos[i + 1], length(secL), length(secR), i)

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
