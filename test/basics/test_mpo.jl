using OpSum.PauliOperators: X, Z
using OpSum
using Test
using VectorInterface: inner

N = 5
vertices = 1:N

g = 1.0
J = 1.0
ops_onesite = sum(vertices) do i
    return -g * Z[i]
end
ops_twosite = sum(vertices[1:(end - 1)]) do i
    return J * X[i] * X[i + 1]
end
ops = ops_onesite + ops_twosite

Ws, Ms = OpSum.opsum_vertex_operators(vertices, ops)
ex = OpSum.mpo_to_opsum(Ws)

for (i, W) in enumerate(Ws)
    @info "W[$i] =" W
end


ex == ops

