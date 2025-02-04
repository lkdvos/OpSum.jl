"""
    opsum_state_machine(opsum)

Convert a representation of a sum of operators into a finite state machine representation.
"""
function opsum_state_machine(opsum::DAWGDictionary)
    opsum_keys = keys(opsum)
    chain_length = depth(opsum)
    T = eltype(opsum) # coefficient_type

    # preallocate storage
    state_machines = Dictionary[] # TODO: fix type
    current_prefixes = [empty_prefix(opsum_keys)]
    next_prefixes = similar(current_prefixes, 0)

    for site in 1:chain_length
        println()
        W = Dictionary() # TODO: fix type
        register = state_registers(opsum_keys, site)

        for (row, prefix) in enumerate(current_prefixes)
            current_state = partial_getindex(opsum_keys, prefix)
            for (k, next_state) in pairs(children(current_state))
                next_prefix = vcat(prefix, k)

                # find column index of the suffix:
                # loop through children that come before and add to offset
                col = 0
                for (s, state) in enumerate(register)
                    state == next_state && break
                    if s == 1 # starting state has many suffixes, but counts as 1
                        col += 1
                    else
                        col += length(state)
                    end
                end

                # loop through suffixes and add all transitions
                if interaction_started(next_prefix)
                    for suffix in next_state
                        col += 1
                        # include coefficient at start of interaction
                        coefficient = row == 1 ? opsum[vcat(next_prefix, suffix)] : one(T)
                        insert!(W, CartesianIndex(row, col), k => coefficient)
                        @info "W[$row, $col] += $coefficient * $k"
                    end
                else
                    # special case interaction hasn't started: counts as single state
                    col = 1
                    coefficient = one(T)
                    @info "W[$row, $col] += $coefficient * $k"
                    insert!(W, CartesianIndex(row, col), k => coefficient)
                end

                # add to list of next prefixes to treat if interaction did not end
                interaction_ended(next_state) || push!(next_prefixes, next_prefix)
            end
        end

        push!(state_machines, W)

        # add ending state back in
        if site != 1
            row = length(current_prefixes) + 1
            col = length(next_prefixes) + 1
            insert!(
                W,
                CartesianIndex(row, col),
                last(instances(eltype(keytype(opsum)))) => one(T),
            )
        end
        # push!(next_prefixes, prefix_interaction_ended(opsum_keys, site))

        # reuse memory for storing prefixes
        current_prefixes, next_prefixes = next_prefixes, current_prefixes
        empty!(next_prefixes)
    end

    return state_machines
end

"""
    interaction_started(prefix)

Determine if an operator string has already acted non-trivially.

See also [`isbegin`](@ref).
"""
interaction_started(prefix) = !isbegin(last(prefix))

"""
    interaction_ended(state)

Determine if an operator string will no longer act non-trivially.

See also [`isend`](@ref).
"""
interaction_ended(state::SDAWG) =
    length(state.children) == 1 && isend(only(keys(state.children)))

function prefix_interaction_ended(inds::SDAWGIndices, site::Int)
    return fill(last(instances(eltype(keytype(inds)))), site)
end
