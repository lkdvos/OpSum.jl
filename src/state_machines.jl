"""
    opsum_state_machine(opsum)

Convert a representation of a sum of operators into a finite state machine representation.
"""
function opsum_state_machine(opsum::DAWGDictionary)
    opsum_keys = keys(opsum)
    chain_length = depth(opsum)
    T = eltype(opsum) # coefficient_type
    TW = operator_type(eltype(keytype(opsum)), T) # single operator type

    # preallocate storage
    vertex_operators = SparseMatrixDOK{TW}[]
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
        W = Dictionary{CartesianIndex{2},TW}()
        current_prefixes = next_prefixes
        next_prefixes = similar(current_prefixes, 0)

        # starting operators treated separately:
        # they need to include the coefficient
        row = 1
        current_state = first(current_register)
        for (k, next_state) in pairs(children(current_state))
            site == 1 && isend(k) && continue
            site == chain_length && isbegin(k) && continue

            if isbegin(k)
                # handle this separately, only single suffix counts
                insert!(W, CartesianIndex(row, 1), TW(k))
                continue
            end

            col = state_offset(next_register, next_state)

            # no next state on last site
            site == chain_length && (col -= 1)

            # loop over all suffixes and add
            for suffix in next_state
                col += 1
                key = vcat(@view(trivial_prefix[1:(site - 1)]), k, suffix)
                increaseindex!(W, CartesianIndex(row, col), opsum[key] * TW(k))

                if !(isempty(suffix) || isend(suffix[1]))
                    push!(next_prefixes, vcat(trivial_prefix[1:(site - 1)], k))
                end
            end
        end

        # add other operators
        for current_state in @view(current_register[2:end])
            for (k, next_state) in pairs(children(current_state))
                @assert !isbegin(k)

                col = state_offset(next_register, next_state)
                site == chain_length && (col -= 1)

                # loop over all suffixes and add
                for j in 1:length(next_state)
                    insert!(W, CartesianIndex(row + j, col + j), TW(k))
                end
                row += length(next_state)
            end
        end

        push!(vertex_operators, _instantiate_matrix(W))

        # add bond coefficients
        if site != 1
            M = Dictionary{CartesianIndex{2},T}()
            for (row, prefix) in enumerate(current_prefixes)
                state = partial_getindex(opsum_keys, prefix)
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
            @assert size(last(bond_coefficients), 2) + 2 == size(last(vertex_operators), 1) "incompatible sizes"
            @debug "coefficients left of $site" M = last(bond_coefficients)
        end
        @debug "operators at site $site" W = last(vertex_operators)
    end

    return vertex_operators, bond_coefficients
end

# state_offset counts all states that come before a given state.
# the first state may be interpreted as "not-started", and thus has length 1
function state_offset(register, state)
    @assert state in register "should not happen"
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

function increaseindex!(W, index, val)
    if haskey(W, index)
        W[index] += val
    else
        insert!(W, index, val)
    end
end

function _instantiate_matrix(W)
    nrows, ncols = mapreduce(Tuple, (x, y) -> max.(x, y), keys(W); init=(1, 1))
    Wmat = SparseArrayDOK{eltype(W)}(undef, (nrows, ncols))
    for (I, v) in pairs(W)
        Wmat[I] = v
    end
    return Wmat
end
