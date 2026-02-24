"""
    TreeTopology(n, edges; root=1)

Tree topology for TTNO construction. `edges` are undirected pairs `(u, v)` with 1-based
site indices. The tree is rooted at `root` and stores parent/children, BFS levels,
and subtree masks for fast edge cuts.
"""
struct TreeTopology
    n::Int
    root::Int
    parent::Vector{Int}
    children::Vector{Vector{Int}}
    levels::Vector{Vector{Int}}
    subtree_mask::Vector{BitVector}
    edge_ids::Dict{Tuple{Int, Int}, Int}
    directed_edges::Vector{Tuple{Int, Int}}
end

function TreeTopology(n::Int, edges::Vector{Tuple{Int, Int}}; root::Int = 1)
    @assert 1 <= root <= n
    adj = [Int[] for _ in 1:n]
    for (u, v) in edges
        @assert 1 <= u <= n && 1 <= v <= n && u != v
        push!(adj[u], v)
        push!(adj[v], u)
    end

    parent = zeros(Int, n)
    children = [Int[] for _ in 1:n]
    levels = Vector{Vector{Int}}()

    # BFS to set parents, children, and levels
    visited = falses(n)
    queue = [root]
    visited[root] = true
    while !isempty(queue)
        lvl = Int[]
        nextq = Int[]
        for u in queue
            push!(lvl, u)
            for v in adj[u]
                if !visited[v]
                    visited[v] = true
                    parent[v] = u
                    push!(children[u], v)
                    push!(nextq, v)
                end
            end
        end
        push!(levels, lvl)
        queue = nextq
    end
    @assert all(visited) "TreeTopology expects a connected tree."

    # Post-order to build subtree masks
    subtree_mask = [falses(n) for _ in 1:n]
    for u in reverse(vcat(levels...))
        mask = subtree_mask[u]
        mask[u] = true
        for v in children[u]
            mask .|= subtree_mask[v]
        end
    end

    # Directed edge ids
    edge_ids = Dict{Tuple{Int, Int}, Int}()
    directed_edges = Tuple{Int, Int}[]
    for (u, v) in edges
        push!(directed_edges, (u, v))
        edge_ids[(u, v)] = length(directed_edges)
        push!(directed_edges, (v, u))
        edge_ids[(v, u)] = length(directed_edges)
    end

    return TreeTopology(n, root, parent, children, levels, subtree_mask, edge_ids, directed_edges)
end

"""
    TTNOTerm(coeff, ops)

Represents a single operator-string term for TTNO construction.
`ops` is a length-`n` vector of local operators (including identity).
"""
struct TTNOTerm{T, Op}
    coeff::T
    ops::Vector{Op}
end

"""
    ttno_terms(vertices, ex::GlobalOp) -> Vector{TTNOTerm}

Convert a `GlobalOp` into TTNO terms (coeff + operator string).
"""
function ttno_terms(vertices::AbstractVector{Int}, ex::GlobalOp)
    coeffs, opstrings = operatorstrings(vertices, ex)
    return [TTNOTerm(c, o) for (c, o) in zip(coeffs, opstrings)]
end

struct SubtreeSig{Op}
    hash::UInt64
    ops::Vector{Tuple{Int, Op}}
end

struct SigIntern{Op}
    sigs::Vector{SubtreeSig{Op}}
    buckets::Dict{UInt64, Vector{Int}}
end

SigIntern{Op}() where {Op} = SigIntern{Op}(SubtreeSig{Op}[], Dict{UInt64, Vector{Int}}())

function subtree_signature(term::TTNOTerm{T, Op}, sites::Vector{Int}) where {T, Op}
    ops = Tuple{Int, Op}[]
    for i in sites
        op = term.ops[i]
        isone(op) && continue
        push!(ops, (i, op))
    end
    sort!(ops, by = first)
    h = hash(length(ops))
    for (i, op) in ops
        h = hash((i, op), h)
    end
    return SubtreeSig{Op}(h, ops)
end

function intern_signature!(intern::SigIntern{Op}, sig::SubtreeSig{Op}) where {Op}
    inds = get!(intern.buckets, sig.hash) do
        Int[]
    end
    for idx in inds
        intern.sigs[idx].ops == sig.ops && return idx
    end
    push!(intern.sigs, sig)
    push!(inds, length(intern.sigs))
    return length(intern.sigs)
end

struct BondUV{T, Op}
    parent::Int
    child::Int
    U::Vector{SubtreeSig{Op}}
    V::Vector{SubtreeSig{Op}}
    gamma::Matrix{T}
    coverU::BitVector
    coverV::BitVector
end

"""
    ttno_bond_optimizations(tree, terms) -> Vector{BondUV}

Compute U/V sets and bond optimizations for each tree edge (parent -> child).
This follows the TTNO double-pass logic at the per-bond level: it builds the
non-redundant U/V sets, constructs the gamma matrix, and computes a minimum
vertex cover via Hopcroft–Karp.
"""
function ttno_bond_optimizations(tree::TreeTopology, terms::Vector{TTNOTerm})
    n = tree.n
    @assert !isempty(terms) "ttno_bond_optimizations requires at least one term."
    nodes = collect(1:n)
    results = BondUV[]

    for u in nodes
        for v in tree.children[u]
            # U = subtree at v, V = complement
            maskU = tree.subtree_mask[v]
            sitesU = findall(maskU)
            sitesV = findall(.!maskU)

            opT = eltype(first(terms).ops)
            internU = SigIntern{opT}()
            internV = SigIntern{opT}()

            u_index = Vector{Int}(undef, length(terms))
            v_index = Vector{Int}(undef, length(terms))
            for (i, term) in enumerate(terms)
                sigU = subtree_signature(term, sitesU)
                sigV = subtree_signature(term, sitesV)
                u_index[i] = intern_signature!(internU, sigU)
                v_index[i] = intern_signature!(internV, sigV)
            end

            nU = length(internU.sigs)
            nV = length(internV.sigs)
            T = typeof(first(terms).coeff)
            gamma = zeros(T, nU, nV)
            for (i, term) in enumerate(terms)
                gamma[u_index[i], v_index[i]] += term.coeff
            end

            adjacency = gamma .!= 0
            coverU, coverV, _, _, _ = min_vertex_cover_bipartite(adjacency)

            push!(results, BondUV(u, v, internU.sigs, internV.sigs, gamma, coverU, coverV))
        end
    end

    return results
end
