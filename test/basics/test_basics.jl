using OpSum
using Test: @test, @testset
using OpSum: depth, opsum_state_machine, compress_state_machine

N = 6

using OpSum.PauliOperators: B, I, X, Y, Z, E

ops_onesite = map(1:N) do i
    return vcat(fill(B, i - 1), [Z], fill(E, N - i))
end
ops_twosite = map(1:(N - 1)) do i
    return vcat(fill(B, i - 1), [X, X], fill(E, N - i - 1))
end

list = vcat(ops_onesite, ops_twosite)
dawg = DAWGDictionary(sort!(list), collect(1:length(list)))

Ws, = opsum_state_machine(dawg)
# Ws = compress_state_machine(Ws)
@info "simple"
foreach(display, Ws)

## next-nearest neighbour:
ops_nnn = map(1:(N - 2)) do i
    return vcat(fill(B, i - 1), [X, I, X], fill(E, N - i - 2))
end
list = sort(vcat(ops_onesite, ops_twosite, ops_nnn, [fill(B, N), fill(E, N)]))
dawg = DAWGDictionary(list, trues(length(list)))

Ws, Ms = opsum_state_machine(dawg);

ops_all_to_all = Vector{OpSum.PauliOperators.PauliBasis}[fill(B, N), fill(E, N)]

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
    ops_all_to_all, vcat([0.0], ones(Float64, length(ops_all_to_all)), [0.0])
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
