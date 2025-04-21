"""
    opsum_vertex_operators(opsum) -> Ws::Vector{<:SparseMatrixDOK}

Convert a representation of a sum of operators into a finite state machine representation.
"""
function opsum_vertex_operators(opsum::DawgDictionary)
    opsum_keys = keys(opsum)
    chain_length = depth(opsum)
    T = eltype(opsum) # coefficient_type
    TW = eltype(keytype(opsum)) # single operator type
    trivial_prefix = first(opsum_keys)

    vertices = 1:chain_length
    return map(vertices) do site
        # initialize variables
        W = Dictionary{CartesianIndex{2},TW}()
        current_register = state_registers(opsum_keys, site - 1)
        next_register = state_registers(opsum_keys, site)

        row = 1
        isstarting = true
        for current_state in current_register # loop over all states
            for (k, next_state) in pairs(children(current_state)) # loop over transitions
                @info "moving" site k current_state next_state
                # handle special cases first:
                (site == first(vertices) && isend(k)) ||
                    (site == last(vertices) && isbegin(k)) && continue

                if isbegin(k) # handle this separately, only single suffix counts
                    insert!(W, CartesianIndex(row, 1), k)
                    continue
                end

                # determine how many "states" are before the current state
                col = state_offset(next_register, next_state)
                site == chain_length && (col -= 1)

                # loop over all suffixes and add
                if isstarting # coefficients are only added at "first" site of operator
                    for suffix in next_state
                        col += 1
                        key = vcat(@view(trivial_prefix[1:(site - 1)]), k, suffix)
                        setwith!(+, W, CartesianIndex(row, col), opsum[key] * k)
                    end
                else
                    for j in 1:length(next_state)
                        insert!(W, CartesianIndex(row + j, col + j), k)
                    end
                    row += length(next_state)
                end
            end
            isstarting = false
        end

        W_mat = _instantiate_matrix(W)
        @debug "operators at site $site" W = W_mat
        return W_mat
    end
end

opsum_vertex_operators(vertices, ex::GlobalOp) = opsum_vertex_operators(Trie(vertices, ex))

function opsum_vertex_operators(opsum::Trie)
    vertices = 1:depth(opsum)

    TW = keytype(opsum) # single operator type
    indices = [
        CartesianIndex{2}[CartesianIndex(1, 1), CartesianIndex(2, 2)] for _ in vertices
    ]
    operators = [TW[begin_marker(TW), end_marker(TW)] for _ in vertices]

    # recursive depth first search
    lvl = 1
    for p in pairs(opsum.children)
        _opsum_vertex_operators!(@view(indices[1:end]), @view(operators[1:end]), p, lvl)
    end

    return map(indices, operators) do inds, ops
        nrows, ncols = mapreduce(Tuple, (x, y) -> max.(x, y), inds; init=(1, 1))
        W = SparseMatrixDOK{TW}(undef, (nrows, ncols))
        for (I, v) in zip(inds, ops)
            row = if I[1] == 1
                1
            elseif I[1] == 2
                nrows
            else
                I[1] - 1
            end
            col = if I[2] == 1
                1
            elseif I[2] == 2
                ncols
            else
                I[2] - 1
            end
            W[row, col] = v
        end
        return W
    end
end

function _opsum_vertex_operators!(indices, operators, (op, child), lvl)
    lvl == 2 && return nothing # early bailout if interaction ended

    nextlvl = if isbegin(op)
        1
    elseif isend(child)
        2
    else
        # TODO: might not need to search all values - last one should be max
        maximum(Base.Fix2(getindex, 2), indices[1]; init=2) + 1
    end

    # add the operator to the list
    if !(lvl == nextlvl == 2)
        push!(first(indices), CartesianIndex(lvl, nextlvl))
        push!(first(operators), child.value)
    end

    nextindices = @view(indices[2:end])
    nextoperators = @view(operators[2:end])
    for p in pairs(child.children)
        _opsum_vertex_operators!(nextindices, nextoperators, p, nextlvl)
    end

    return nothing
end

function QuantumOperatorAlgebra.isend(op::Trie)
    return isend(op.value) ||
           isempty(op.children) ||
           (length(op.children) == 1 && isend(first(keys(op.children))))
end

function mpo_to_opsum(Ws::Vector{<:SparseMatrixDOK{<:LocalOp}})
    vertices = eachindex(Ws)
    globalops = map(vertices) do i
        return map(Array(Ws[i])) do W
            w = (isend(W) || isbegin(W)) ? one(W) : W
            return GlobalOp(SiteOp(w, [i]))
        end
    end
    return simplify(reduce(*, globalops)[1, end])
end

function opsum_bond_coefficients(opsum::DawgDictionary)
    opsum_keys = keys(opsum)
    chain_length = depth(opsum)
    T = eltype(opsum) # coefficient_type

    # preallocate storage
    bond_coefficients = SparseMatrixDOK{T}[]
    trivial_prefix = first(opsum_keys)
    next_register = state_registers(opsum_keys, 0)
    next_prefixes = map(
        x -> [x], filter!(!isbegin, collect(keys(children(root(opsum_keys)))))
    )

    for site in 1:chain_length
        # initialize variables
        current_register = next_register
        next_register = state_registers(opsum_keys, site)
        current_prefixes = next_prefixes
        next_prefixes = similar(current_prefixes, 0)

        # Collect the prefixes for the next site
        row = 1
        current_state = first(current_register)
        for (k, next_state) in pairs(children(current_state))
            (isbegin(k) || isend(k)) && continue
            for suffix in next_state
                if !isempty(suffix) && !isend(suffix[1])
                    push!(next_prefixes, vcat(trivial_prefix[1:(site - 1)], k))
                end
            end
        end

        site == 1 && continue

        # generate bond coefficients
        M = Dictionary{CartesianIndex{2},T}()
        for (row, prefix) in enumerate(@show unique!(current_prefixes))
            @show state = partial_getindex(opsum_keys, prefix)
            col = state_offset(current_register, state)
            col -= 2 # skip first two columns: no starting/ending state
            for suffix in state
                (isbegin(first(suffix)) || isend(first(suffix))) && continue
                key = vcat(prefix, suffix)
                coefficient = opsum[key]
                col += 1
                insert!(M, CartesianIndex(row, col), coefficient)
                if !interaction_ended(suffix)
                    push!(next_prefixes, vcat(prefix, suffix[1:1]))
                end
            end
        end
        push!(bond_coefficients, _instantiate_matrix(M))

        @debug "coefficients left of $site" M = last(bond_coefficients)
    end

    return bond_coefficients
end

# state_offset counts all states that come before a given state.
# the first state may be interpreted as "not-started", and thus has length 1
function state_offset(register, state)
    @assert state in register "should not happen:\n$state\n\n ∉\n\n $register"
    offset = 1
    state == first(register) && return offset
    for state′ in @view(register[2:end])
        state == state′ && break
        offset += length(state′)
    end
    return offset
end

"""
    interaction_ended(state)

Determine if an operator string will no longer act non-trivially.

See also [`isend`](@ref).
"""
interaction_ended(suffix) = length(suffix) == 1 || isend(suffix[2])

function _instantiate_matrix(W)
    nrows, ncols = mapreduce(Tuple, (x, y) -> max.(x, y), keys(W); init=(1, 1))
    Wmat = SparseArrayDOK{eltype(W)}(undef, (nrows, ncols))
    for (I, v) in pairs(W)
        Wmat[I] = v
    end
    return Wmat
end

#=
        [1 D C]   [1 . .]
W′ =    [. 1 .] * [. 1 .]
        [. B A]   [. . V]

=#
function right_multiply(W, V)
    V_extended = cat([one(eltype(V))], [one(eltype(V))], V; dims=(1, 2))
    return W * V_extended
end

function left_multiply(Vᴴ, W)
    Vᴴ_extended = cat([one(eltype(Vᴴ))], [one(eltype(Vᴴ))], Vᴴ; dims=(1, 2))
    return Vᴴ_extended * W
end
