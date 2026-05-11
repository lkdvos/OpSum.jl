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
        W = Dictionary{CartesianIndex{2}, TW}()
        current_register = state_registers(opsum_keys, site - 1)
        next_register = state_registers(opsum_keys, site)

        row = 1
        isstarting = true
        for current_state in current_register # loop over all states
            for (k, next_state) in pairs(children(current_state)) # loop over transitions
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

opsum_vertex_operators(vertices, ex::GlobalOp) = opsum_vertex_operators(vertices, Trie(vertices, ex))

function opsum_vertex_operators(vertices, opsum::Trie)
    dawgdict = DawgDictionary(opsum)

    T = valtype(opsum) # coefficient type
    TW = LocalOp{T, eltype(keytype(opsum))} # single operator type
    vertex_operators = [
        Pair{CartesianIndex{2}, TW}[
                CartesianIndex(1, 1) => one(TW), CartesianIndex(2, 2) => one(TW),
            ] for _ in vertices
    ]
    bond_coefficients = [
        Pair{CartesianIndex{2}, T}[
                CartesianIndex(1, 1) => one(T), CartesianIndex(2, 2) => one(T),
            ] for _ in vertices[1:(end - 1)]
    ]

    # recursive depth first search to place all operators
    lvl = 1
    prefix = keytype(opsum)()
    for p in pairs(opsum.children)
        _opsum_vertex_operators!(
            @view(vertex_operators[1:end]),
            @view(bond_coefficients[1:end]),
            p,
            lvl,
            prefix,
            dawgdict,
        )
    end

    Ws = map(vertex_operators) do vertex_operator
        nrows, ncols = mapreduce(
            Tuple ∘ first, (x, y) -> max.(x, y), vertex_operator; init = (1, 1)
        )
        W = SparseMatrixDOK{TW}(undef, (nrows, ncols))
        for (I, v) in vertex_operator
            row = I[1] == 1 ? 1 : I[1] == 2 ? nrows : I[1] - 1
            col = I[2] == 1 ? 1 : I[2] == 2 ? ncols : I[2] - 1
            W[row, col] = v
        end
        return W
    end

    Ms = map(enumerate(bond_coefficients)) do (i, bond_coefficient)
        nrows, ncols = mapreduce(
            Tuple ∘ first, (x, y) -> max.(x, y), bond_coefficient; init = (3, 3)
        )
        # ncols = size(Ws[i + 1], 1)
        M = SparseMatrixDOK{T}(undef, (nrows, ncols))
        for (I, v) in bond_coefficient
            row = I[1] == 1 ? 1 : I[1] == 2 ? nrows : I[1] - 1
            col = I[2] == 1 ? 1 : I[2] == 2 ? ncols : I[2] - 1
            M[row, col] = v
        end
        return M
    end

    return Ws, Ms
end

function _opsum_vertex_operators!(
        vertex_operators, bond_coefficients, (op, child), lvl, prefix, dawgdict
    )
    lvl == 2 && return nothing # early bailout if interaction ended

    nextlvl = if lvl == 1 && isone(op)
        1
    elseif isend(child)
        2
    else
        # TODO: might not need to search all values - last one should be max
        maximum(Base.Fix2(getindex, 2) ∘ first, vertex_operators[1]; init = 2) + 1
    end

    # add the operator to the list
    if !(lvl == nextlvl == 2)
        ind = CartesianIndex(lvl, nextlvl)
        op′ = isnothing(child.value) ? op : child.value * op
        push!(first(vertex_operators), ind => op′)
    end

    # add the coefficients to the list
    if !isempty(prefix) && nextlvl != 1 && nextlvl != 2
        state = partial_getindex(keys(dawgdict), prefix)
        register = state_registers(keys(dawgdict), length(prefix))
        col = state_offset(register, state)
        for suffix in state
            interaction_ended(suffix) && continue
            key = vcat(prefix, suffix)
            coefficient = dawgdict[key]
            col += 1
            push!(first(bond_coefficients), CartesianIndex(nextlvl, col) => coefficient)
        end
    end

    next_vertex_operators = @view(vertex_operators[2:end])
    next_bond_coefficients =
        isempty(prefix) ? @view(bond_coefficients[1:end]) : @view(bond_coefficients[2:end])

    nextprefix = vcat(prefix, op)
    for (op, child) in pairs(child.children)

        _opsum_vertex_operators!(
            next_vertex_operators, next_bond_coefficients,
            (op, child),
            nextlvl, nextprefix, dawgdict,
        )
    end

    return nothing
end

function opsum_bond_coefficients(opsum::Trie)
    dawgdict = DawgDictionary(opsum)
    vertices = 1:depth(opsum)

    TW = valtype(opsum) # coefficient type
    indices = [
        CartesianIndex{2}[CartesianIndex(1, 1), CartesianIndex(2, 2)] for _ in vertices
    ]
    operators = [TW[one(TW), one(TW)] for _ in vertices]

    # recursive depth first search
    lvl = 1
    prefix = keytype(opsum)()
    for (op, child) in pairs(opsum.children)
        _opsum_bond_coefficients!(
            @view(indices[1:end]), @view(operators[1:end]), p, prefix, lvl, dawgdict
        )
    end
    return
end
function _opsum_bond_coefficients!(indices, operators, (op, child), prefix, lvl, dawgdict)
    lvl == 2 && return nothing # early bailout if interaction ended

    nextlvl = if isbegin(op)
        1
    elseif isend(child)
        2
    else
        # TODO: might not need to search all values - last one should be max
        maximum(Base.Fix2(getindex, 2), indices[1]; init = 2) + 1
    end

    # add coefficient to the list
    return if lvl != 1 && nextlvl != 1 && nextlvl != 2
        state = partial_getindex(keys(dawgdict), prefix)
        col = state_offset(register, state)
        for suffix in state
            (isbegin(first(suffix)) || isend(first(suffix))) && continue
            key = vcat(prefix, suffix)
            coefficient = dawgdict[key]
            col += 1
            push!(first(indices), CartesianIndex(lvl, col))
            push!(first(operators), coefficient)
        end
    end
end

# function QuantumOperatorAlgebra.isend(op::Trie)
#     return !isnothing(op.value) ||
#         isempty(op.children) ||
#         (length(op.children) == 1 && isend(first(keys(op.children))))
# end

function mpo_to_dense(Ws::Vector{<:SparseMatrixDOK{<:LocalOp}}, sites)
    N = length(Ws)
    T = ComplexF64

    # Instantiate W[1] entries into physical-space matrices
    D_left, D_right = size(Ws[1])
    d = sites[1]
    result = Matrix{Matrix{T}}(undef, D_left, D_right)
    for r in 1:D_left, c in 1:D_right
        result[r, c] = zeros(T, d, d)
    end
    for (idx, v) in storedpairs(Ws[1])
        result[idx[1], idx[2]] = T.(instantiate(v, sites[1]))
    end

    # Contract left-to-right via matrix multiplication with kron
    for i in 2:N
        D_old_right = size(result, 2)
        D_new_right = size(Ws[i], 2)
        di = sites[i]

        # Instantiate W[i]
        Wi = Matrix{Matrix{T}}(undef, D_old_right, D_new_right)
        for r in 1:D_old_right, c in 1:D_new_right
            Wi[r, c] = zeros(T, di, di)
        end
        for (idx, v) in storedpairs(Ws[i])
            Wi[idx[1], idx[2]] = T.(instantiate(v, sites[i]))
        end

        # Matrix multiply: new[r,c] = sum_k kron(result[r,k], Wi[k,c])
        phys_dim = size(result[1, 1], 1) * di
        D_old_left = size(result, 1)
        new_result = Matrix{Matrix{T}}(undef, D_old_left, D_new_right)
        for r in 1:D_old_left, c in 1:D_new_right
            new_result[r, c] = zeros(T, phys_dim, phys_dim)
            for k in 1:D_old_right
                new_result[r, c] .+= kron(result[r, k], Wi[k, c])
            end
        end
        result = new_result
    end

    return result[1, end]
end

function mpo_to_opsum(Ws::Vector{<:SparseMatrixDOK{<:LocalOp}})
    vertices = eachindex(Ws)
    globalops = map(vertices) do i
        return map(Array(Ws[i])) do W
            # w = (isend(W) || isbegin(W)) ? one(W) : W
            return GlobalOp(SiteOp(W, [i]))
        end
    end
    return reduce(*, globalops)[1, end]
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
        M = Dictionary{CartesianIndex{2}, T}()
        for (row, prefix) in enumerate(unique!(current_prefixes))
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
"""
interaction_ended(suffix) = length(suffix) == 1 || isend(suffix[2])
interaction_ended(suffix::Vector{<:OperatorBasis}) = all(isone, suffix)

function _instantiate_matrix(W)
    nrows, ncols = mapreduce(Tuple, (x, y) -> max.(x, y), keys(W); init = (1, 1))
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
    V_extended = cat([one(eltype(V))], [one(eltype(V))], V; dims = (1, 2))
    return W * V_extended
end

function left_multiply(Vᴴ, W)
    Vᴴ_extended = cat([one(eltype(Vᴴ))], [one(eltype(Vᴴ))], Vᴴ; dims = (1, 2))
    return Vᴴ_extended * W
end
