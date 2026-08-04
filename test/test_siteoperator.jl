using Test
using OpSum
using OpSum: SiteOperator, scalarop, prune, instantiate, passthrough, ispassthrough
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using TensorKit: sectortype
using VectorInterface: scalartype, scale, add, inner
using LinearAlgebra: norm

# `SiteOperator` is a small enough type to test directly, and worth it: it is the element type of the
# reduced bond matrices as well as the on-site operand type, so the sweep leans on its arithmetic
# (`+` accumulating onto a matching letter is exactly what happens when two uncovered left vertices
# fold into one bond index) and on `prune` (a bond entry carrying an explicit zero would spend a bond
# index on nothing). Exercising it through the pipeline alone leaves that surface incidental.

const V = SU2Space(1 // 2 => 1)
const A = IrrepOperator{SU2Irrep}(SU2Irrep(0), 1)
const B = IrrepOperator{SU2Irrep}(SU2Irrep(1), 1)

@testset "construction and conversion" begin
    @test length(SiteOperator{SU2Irrep}()) == 0
    @test isempty(SiteOperator{SU2Irrep}())

    # a bare letter promotes with unit coefficient
    @test collect(pairs(SiteOperator(A))) == [A => ComplexF64(1)]
    @test convert(SiteOperator{SU2Irrep}, A) == SiteOperator(A)

    # a scalar is the pass-through sentinel letter, not a separate representation
    s = SiteOperator{SU2Irrep}(2.5)
    @test only(keys(s)) == passthrough(SU2Irrep)
    @test ispassthrough(only(keys(s)))
    @test only(values(s)) == ComplexF64(2.5)
    @test convert(SiteOperator{SU2Irrep}, 2.5) == s
    @test scalarop(2.5, V) == s
    @test scalarop(2.5, SU2Irrep) == s

    # ...and a zero scalar is the *empty* operator, so it carries no letter at all
    @test isempty(SiteOperator{SU2Irrep}(0))

    @test_throws AssertionError SiteOperator{SU2Irrep}([A], ComplexF64[1, 2])
end

@testset "traits" begin
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

    # `iszero` is about the coefficients; an operator can be zero while still carrying letters, which
    # is exactly the state `prune` exists to clean up. (Note `A - A` on two bare letters does not
    # promote — `+` and scalar `*` do, `-` does not — so this goes through `SiteOperator` explicitly.)
    z = SiteOperator(A) - SiteOperator(A)
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

    # `isone` means the bare identity with unit coefficient — i.e. exactly the pass-through channel
    @test isone(one(SiteOperator{SU2Irrep}))
    @test isone(scalarop(1, V))
    @test !isone(scalarop(2, V))            # right letter, wrong coefficient
    @test !isone(SiteOperator(A))           # trivial *charge*, but an enumerated letter, not passthrough
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
    @test keys(A + B) == [A, B]

    @test values(-(2 * A)) == ComplexF64[-2]
    @test 5 * A - 2 * A == 3 * A
    @test values(2 * A * 3) == ComplexF64[6]
    @test 3 * (2 * A) == 6 * A                # scalar on either side
    @test (4 * A) / 2 == 2 * A

    # a bare letter times a scalar promotes, which the sweep uses when weighting a letter by its
    # reduced coefficient
    @test A * 2 == 2 * SiteOperator(A)
    @test 2 * A == SiteOperator(A) * 2
    @test A + B == SiteOperator(A) + SiteOperator(B)

    # the operands are not mutated by any of it
    x, y = 2 * A, 3 * B
    _ = x + y, x - y, -x, 2x, x / 2
    @test x == 2 * A && y == 3 * B
end

@testset "VectorInterface and norm" begin
    @test scale(2 * A, 3) == 6 * A
    @test add(2 * A, 3 * B) == 2 * A + 3 * B

    # the alphabet is orthonormal, so the inner product is the plain coefficient overlap. (`inner` is
    # defined on two `SiteOperator`s; a bare letter would fall through to `LinearAlgebra.dot`.)
    @test inner(2 * A, 3 * A) ≈ 6
    @test inner(SiteOperator(A), SiteOperator(B)) ≈ 0
    @test inner((1 + 2im) * A, SiteOperator(A)) ≈ 1 - 2im   # conjugate-linear in the first argument
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
    @test sprint(show, zero(SiteOperator{SU2Irrep})) == "SiteOperator{SU2Irrep}(0)"
    # a unit coefficient is elided, a non-unit one printed
    @test sprint(show, SiteOperator(A)) == "SiteOperator{SU2Irrep}(" * sprint(show, A) * ")"
    @test occursin("2.0", sprint(show, 2 * A))
    @test occursin(" + ", sprint(show, A + B))
end

@testset "instantiation" begin
    # a scalar materialises structurally as `c * id(V)`, never through `one(::Type)`
    @test instantiate(scalarop(2.0, V), V) ≈ 2 * id(V)
    # and a combination distributes
    @test instantiate(2 * B, V) ≈ 2 * instantiate(B, V)
    @test_throws ArgumentError instantiate(zero(SiteOperator{SU2Irrep}), V)
end
