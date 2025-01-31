using Test
using OpSum
using Dictionaries

vals = unique!(
    sort!(collect.(vec(collect(Iterators.product(Iterators.repeated(1:4, 3)...)))))
)
d = SDAWG(vals)

@test length(vals) == length(d)
