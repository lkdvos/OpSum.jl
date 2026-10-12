using Test
using OpSum
using OpSum: SiteOperator, scalarop, prune, instantiate, passthrough, ispassthrough
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using TensorKit: sectortype, removeunit
using VectorInterface: scalartype, scale, add, inner
using LinearAlgebra: norm

# `SiteOperator` is the element type of the bond matrices, so the sweep leans on its arithmetic (`+`
# accumulating onto a matching letter) and on `prune` (an explicit zero would spend a bond index).

const V = SU2Space(1 // 2 => 1)
const A = IrrepOperator{SU2Irrep}(SU2Irrep(0), 2)   # index 1 is the identity
const B = IrrepOperator{SU2Irrep}(SU2Irrep(1), 1)

@testset "construction and conversion" begin
    @test isempty(SiteOperator{SU2Irrep}())

    @test collect(pairs(SiteOperator(A))) == [A => ComplexF64(1)]
    @test convert(SiteOperator{SU2Irrep}, A) == SiteOperator(A)

    # a scalar is the pass-through (identity) letter, not a separate representation
    s = SiteOperator{SU2Irrep}(2.5)
    @test only(keys(s)) == passthrough(SU2Irrep)
    @test ispassthrough(only(keys(s)))
    @test only(values(s)) == ComplexF64(2.5)
    @test convert(SiteOperator{SU2Irrep}, 2.5) == s
    @test scalarop(2.5, V) == s
    @test scalarop(2.5, SU2Irrep) == s

    # a zero scalar is the empty operator
    @test isempty(SiteOperator{SU2Irrep}(0))

    @test_throws AssertionError SiteOperator{SU2Irrep}([A], ComplexF64[1, 2])
    @test sectortype(SiteOperator{SU2Irrep}) === SU2Irrep
    @test scalartype(SiteOperator{SU2Irrep}) === ComplexF64
end

@testset "container interface" begin
    op = 2 * A + 3 * B
    @test length(op) == 2
    @test keys(op) == [A, B]
    @test values(op) == ComplexF64[2, 3]
    @test collect(pairs(op)) == [A => ComplexF64(2), B => ComplexF64(3)]
    @test !isempty(op)
    @test !iszero(op)

    # `iszero` is about the coefficients; `prune` removes letters that cancelled
    z = 1 * A - 1 * A
    @test iszero(z)
    @test !isempty(z)
    @test isempty(prune(z))
    @test prune(2 * A + 0 * B) == 2 * A
end

@testset "zero, one, isone" begin
    op = 2 * A
    @test iszero(zero(SiteOperator{SU2Irrep})) && isempty(zero(SiteOperator{SU2Irrep}))
    @test isempty(zero(op))
    @test one(SiteOperator{SU2Irrep}) == SiteOperator{SU2Irrep}(1)
    @test one(op) == SiteOperator{SU2Irrep}(1)

    # `isone`: the pass-through letter with unit coefficient
    @test isone(one(SiteOperator{SU2Irrep}))
    @test isone(scalarop(1, V))
    @test !isone(scalarop(2, V))            # right letter, wrong coefficient
    @test !isone(SiteOperator(A))           # trivial *charge*, but a matrix unit, not the identity
    @test !isone(scalarop(1, V) + SiteOperator(A))
    @test !isone(zero(SiteOperator{SU2Irrep}))
end

@testset "equality and hashing" begin
    @test 2 * A == 2 * A
    @test 2 * A != 3 * A
    @test 2 * A != 2 * B
    @test hash(2 * A) == hash(2 * A)
    @test hash(2 * A) != hash(3 * A)
    # usable as a dictionary key, which the bond-entry accumulation relies on
    d = Dict(2 * A => :x)
    @test d[2 * A] === :x
end

@testset "arithmetic" begin
    # `+` accumulates onto a matching letter rather than appending a duplicate
    @test keys(2 * A + 3 * A) == [A]
    @test values(2 * A + 3 * A) == ComplexF64[5]
    @test keys(SiteOperator(A) + SiteOperator(B)) == [A, B]

    @test values(-(2 * A)) == ComplexF64[-2]
    @test 5 * A - 2 * A == 3 * A
    @test values(2 * A * 3) == ComplexF64[6]
    @test 3 * (2 * A) == 6 * A                # scalar on either side
    @test (4 * A) / 2 == 2 * A

    # a bare letter times a scalar promotes
    @test A * 2 == 2 * SiteOperator(A)
    @test 2 * A == SiteOperator(A) * 2

    # the operands are not mutated by any of it
    x, y = 2 * A, 3 * B
    _ = x + y, x - y, -x, 2x, x / 2
    @test x == 2 * A && y == 3 * B
end

@testset "VectorInterface and norm" begin
    @test scale(2 * A, 3) == 6 * A
    @test add(2 * A, 3 * B) == 2 * A + 3 * B

    # the alphabet is orthonormal, so the inner product is the plain coefficient overlap
    @test inner(2 * A, 3 * A) ≈ 6
    @test inner(A, B) ≈ 0
    @test inner((1 + 2im) * A, A) ≈ 1 - 2im       # conjugate-linear in the first argument
    # a bare letter is accepted on either side, and mixes with a combination
    @test inner(A, 3 * A) ≈ 3
    @test inner(3 * A, A) ≈ 3
    @test inner(A, SiteOperator(A) + SiteOperator(B)) ≈ 1
    @test norm(3 * A) ≈ 3
    @test norm(2 * A + 3 * B) ≈ sqrt(13)
    @test norm(zero(SiteOperator{SU2Irrep})) ≈ 0
end

@testset "isapprox" begin
    @test 2 * A ≈ 2 * A
    @test 2 * A ≈ (2 + 1.0e-14) * A
    @test !(2 * A ≈ 2.1 * A)
    @test !(2 * A ≈ 2 * B)
    # letters that cancelled compare equal to the operator without them
    @test 2 * A + 0 * B ≈ 2 * A
end

@testset "show" begin
    # structure only: the type parameter's qualification depends on what is in scope
    empty_str = sprint(show, zero(SiteOperator{SU2Irrep}))
    @test startswith(empty_str, "SiteOperator{") && endswith(empty_str, "(0)")

    # a unit coefficient is elided, a non-unit one printed
    single = sprint(show, SiteOperator(A))
    @test occursin(sprint(show, A), single)
    @test !occursin("*", single)
    scaled = sprint(show, 2 * A)
    @test occursin("2.0", scaled) && occursin("*", scaled)

    # several letters are joined
    @test occursin(" + ", sprint(show, SiteOperator(A) + SiteOperator(B)))
end

@testset "instantiation" begin
    @test removeunit(instantiate(scalarop(2.0, V), V), 3) ≈ 2 * id(V)
    @test instantiate(2 * B, V) ≈ 2 * instantiate(B, V)
    @test_throws ArgumentError instantiate(zero(SiteOperator{SU2Irrep}), V)
end
