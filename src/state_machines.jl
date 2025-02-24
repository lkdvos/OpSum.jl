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

    # first site
    W = Dictionary{CartesianIndex{2},TW}()
    current_register = state_registers(opsum_keys, 0)
    next_register = state_registers(opsum_keys, 1)
    prefix = first(opsum_keys)

    for (row, current_state) in enumerate(current_register) # should have length 1
        for (k, next_state) in pairs(children(current_state))
            if next_state == first(next_register) # not started
                @assert isbegin(k)
                insert!(W, CartesianIndex(row, 1), TW(k))
                continue
            end

            col_offset = state_offset(next_register, next_state)

            # loop over all suffixes and add
            for (suffix_id, suffix) in enumerate(next_state)
                coefficient = opsum[vcat(k, suffix)]
                # TODO: multiple onsite terms
                insert!(W, CartesianIndex(row, col_offset + suffix_id), coefficient * TW(k))
            end
        end
    end
    push!(vertex_operators, instantiate_W(W))
    @debug "site 1" last(vertex_operators)

    for site in 2:chain_length
        current_register = next_register
        next_register = state_registers(opsum_keys, site)
        W = Dictionary{CartesianIndex{2},TW}()

        # starting operators treated separately: need coefficient
        row_offset = 1
        current_state = first(current_register)
        for (k, next_state) in pairs(children(current_state))
            if isbegin(k)
                @assert site != chain_length
                insert!(W, CartesianIndex(row_offset, 1), TW(k))
                continue
            end

            col_offset = state_offset(next_register, next_state)
            site == chain_length && (col_offset -= 1)

            # loop over all suffixes and add
            for (suffix_id, suffix) in enumerate(next_state)
                key = vcat(@view(prefix[1:(site - 1)]), k, suffix)
                coefficient = opsum[key]
                # TODO: multiple onsite terms
                insert!(
                    W,
                    CartesianIndex(row_offset, col_offset + suffix_id),
                    coefficient * TW(k),
                )
            end
        end

        # rest
        for current_state in @view(current_register[2:end])
            for (k, next_state) in pairs(children(current_state))
                @assert !isbegin(k)

                col_offset = state_offset(next_register, next_state)
                site == chain_length && (col_offset -= 1)

                # loop over all suffixes and add
                for j in 1:length(next_state)
                    insert!(W, CartesianIndex(row_offset + j, col_offset + j), TW(k))
                end
                row_offset += length(next_state)
            end
        end

        push!(vertex_operators, instantiate_W(W))
        @debug "site $site" last(vertex_operators)
    end

    return vertex_operators, bond_coefficients
end

# state_offset counts all states that come before a given state.
# the first state is interpreted as "not-started", and thus has length 1
function state_offset(register, state)
    offset = 1
    state == first(register) && return offset
    for state′ in @view(register[2:end])
        state == state′ && break
        offset += length(state′)
    end
    return offset
end

"""
    interaction_started(prefix)

Determine if an operator string has already acted non-trivially.

See also [`isbegin`](@ref).
"""
interaction_started(prefix) = !isempty(prefix) && any(!isone, prefix)

"""
    interaction_ended(state)

Determine if an operator string will no longer act non-trivially.

See also [`isend`](@ref).
"""
interaction_ended(state::SDAWG) = length(state) == 1 && all(isone, only(state))

# TODO: generalize to non-uniform operators
function prefix_interaction_ended(inds::SDAWGIndices, site::Int)
    return fill(one(eltype(keytype(inds))), site)
end

function instantiate_W(W)
    nrows, ncols = mapreduce(I -> I.I, (x, y) -> max.(x, y), keys(W); init=(1, 1))
    Wmat = SparseArrayDOK{eltype(W)}(undef, (nrows, ncols))
    for (I, v) in pairs(W)
        row = I[1] == -1 ? lastindex(Wmat, 1) : I[1]
        col = I[2] == -1 ? lastindex(Wmat, 2) : I[2]
        Wmat[row, col] = v
    end
    return Wmat
end
function instantiate_M(M)
    nrows, ncols = mapreduce(I -> I.I, (x, y) -> max.(x, y), keys(M); init=(1, 1))
    Mmat = SparseArrayDOK{eltype(M)}(undef, (nrows - 1, ncols))
    for (I, v) in pairs(M)
        @assert I[1] != -1 && I[2] != -1
        row = I[1] == -1 ? lastindex(Mmat, 1) : I[1] - 1
        col = I[2] == -1 ? lastindex(Mmat, 2) : I[2]
        Mmat[row, col] = v
    end
    return Mmat
end
