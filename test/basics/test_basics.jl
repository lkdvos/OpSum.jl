using OpSum
using Test: @test, @testset
using OpSum:
    depth,
    opsum_state_machine,
    compress_state_machine,
    opsum_vertex_operators,
    opsum_bond_coefficients

N = 8
T = Int

using OpSum.PauliOperators: B, I, X, Y, Z, E

ops_onesite = map(1:N) do i
    return vcat(fill(B, i - 1), [Z], fill(E, N - i))
end
ops_twosite = map(1:(N - 1)) do i
    return vcat(fill(B, i - 1), [X, X], fill(E, N - i - 1))
end
ops_bookkeeping = map(0:N) do i
    return vcat(fill(B, i), fill(E, N - i))
end

list = vcat(ops_onesite, ops_twosite, ops_bookkeeping)
dawg = DAWGDictionary(sort!(list), collect(1:length(list)))

Ws = opsum_vertex_operators(dawg)
Ms = opsum_bond_coefficients(dawg)
# Ws = compress_state_machine(Ws)
@info "simple"
foreach(display, Ws)

## next-nearest neighbour:
ops_nnn = map(1:(N - 2)) do i
    return vcat(fill(B, i - 1), [X, I, X], fill(E, N - i - 2))
end

ops_bookkeeping = map(0:N) do i
    return vcat(fill(B, i), fill(E, N - i))
end

list = vcat(ops_onesite, ops_twosite, ops_nnn, ops_bookkeeping)
coeffs = ones(Int, length(list))
coeffs[end .- (1:length(ops_bookkeeping)) .+ 1] .= 0
dawg = DAWGDictionary(list, coeffs)

Ws, Ms = opsum_state_machine(dawg);

ops_all_to_all = Vector{OpSum.PauliOperators.PauliBasis}[ops_bookkeeping...]
for i in 1:N, j in (i + 1):N, k in (j + 1):N, l in (k + 1):N
    ops = fill(I, N)
    ops[1:(i - 1)] .= Ref(B)
    ops[i] = X
    ops[j] = X
    ops[k] = X
    ops[l] = X
    ops[(l + 1):end] .= Ref(E)
    push!(ops_all_to_all, ops)
end

dawg = DAWGDictionary(
    ops_all_to_all,
    vcat(
        zeros(T, length(ops_bookkeeping)),
        ones(T, length(ops_all_to_all) - length(ops_bookkeeping)),
    ),
)

Ws, Ms = opsum_state_machine(dawg)
@info "uncompressed"
io = IOContext(stdout, :limit => false)
println(io, 1)
show(io, MIME"text/plain"(), Ws[1])
println(io)
for i in 1:N
    println(io, i + 1)
    show(io, MIME"text/plain"(), Ms[i])
    println(io)
    show(io, MIME"text/plain"(), Ws[i + 1])
    println(io)
end
foreach(x -> (show(io, MIME"text/plain"(), Matrix(x)); println()), Ws)
@info "Ms:"
foreach(x -> (show(io, MIME"text/plain"(), Matrix(x)); println()), Ms)
# Ws = compress_state_machine(ws)
@info "compressed"
foreach(display, Ws)
