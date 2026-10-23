using Test
using OpSum
using OpSum: SiteOperator, scalarop, instantiate, passthrough, ispassthrough
using OpSum: Letter
using TensorKit
using TensorKit: sectortype, removeunit
using VectorInterface: scalartype, scale, add, inner
using LinearAlgebra: norm

# `SiteOperator` is the entry type of the bond matrices: canonical (letters sorted, duplicates summed, no
# zero coefficients), so `+` accumulates onto a matching letter, cancellation leaves the empty operator and
# `==` is operator equality.

const V = SU2Space(1 // 2 => 1)
const A = Letter{SU2Irrep}(SU2Irrep(0), 2)   # index 1 is the identity
const B = Letter{SU2Irrep}(SU2Irrep(1), 1)
const SO = SiteOperator{SU2Irrep}

@testset "construction, scalars, container interface" begin
    @test isempty(SO()) && isempty(SO(0))
    @test collect(pairs(SiteOperator(A))) == [A => ComplexF64(1)]
    @test convert(SO, A) == SiteOperator(A)
    @test sectortype(SO) === SU2Irrep && scalartype(SO) === ComplexF64
    @test_throws ErrorException SO([A], ComplexF64[1, 2])

    # a scalar is the pass-through (identity) letter
    s = SO(2.5)
    @test only(keys(s)) == passthrough(SU2Irrep) && ispassthrough(only(keys(s)))
    @test only(values(s)) == 2.5
    @test convert(SO, 2.5) == s == scalarop(2.5, sectortype(V)) == scalarop(2.5, SU2Irrep)

    op = 2 * A + 3 * B
    @test length(op) == 2 && keys(op) == [A, B] && values(op) == ComplexF64[2, 3]
    @test collect(pairs(op)) == [A => 2, B => 3]
    z = 1 * A - 1 * A
    @test iszero(z) && isempty(z) && z == zero(SO)
    @test 2 * A + 0 * B == 2 * A
    @test SO([B, A, B], ComplexF64[1, 2, 3]) == 2 * A + 4 * B      # sorted, duplicates summed
    @test SO([A, B], ComplexF64[1, 0]) == SiteOperator(A)
    @test hash(SO([B, A], ComplexF64[3, 2])) == hash(2 * A + 3 * B)
end

@testset "zero, one, isone" begin
    @test iszero(zero(SO)) && isempty(zero(2 * A)) && isempty(zero(SO))
    @test one(SO) == SO(1) == one(2 * A)
    @test isone(one(SO)) && isone(scalarop(1, sectortype(V)))
    @test !isone(scalarop(2, sectortype(V)))
    @test !isone(SiteOperator(A))           # trivial charge, but a matrix unit, not the identity
    @test !isone(scalarop(1, sectortype(V)) + SiteOperator(A)) && !isone(zero(SO))
end

@testset "equality, hashing, arithmetic" begin
    @test 2 * A == 2 * A && 2 * A != 3 * A && 2 * A != 2 * B
    @test hash(2 * A) == hash(2 * A) && hash(2 * A) != hash(3 * A)
    @test Dict(2 * A => :x)[2 * A] === :x

    @test keys(2 * A + 3 * A) == [A] && values(2 * A + 3 * A) == ComplexF64[5]
    @test keys(SiteOperator(A) + SiteOperator(B)) == [A, B]
    @test values(-(2 * A)) == ComplexF64[-2]
    @test 5 * A - 2 * A == 3 * A
    @test 3 * (2 * A) == 6 * A == 2 * A * 3
    @test (4 * A) / 2 == 2 * A
    @test A * 2 == 2 * SiteOperator(A)

    x, y = 2 * A, 3 * B
    _ = x + y, x - y, -x, 2x, x / 2
    @test x == 2 * A && y == 3 * B                  # operands are not mutated

    @test 2 * A ≈ (2 + 1.0e-14) * A
    @test !(2 * A ≈ 2.1 * A) && !(2 * A ≈ 2 * B)
    @test 2 * A + 0 * B ≈ 2 * A
end

@testset "VectorInterface and norm" begin
    @test scale(2 * A, 3) == 6 * A
    @test add(2 * A, 3 * B) == 2 * A + 3 * B
    # the alphabet is orthonormal; a bare letter is accepted on either side
    @test inner(2 * A, 3 * A) ≈ 6 && inner(A, B) ≈ 0
    @test inner((1 + 2im) * A, A) ≈ 1 - 2im       # conjugate-linear in the first argument
    @test inner(A, 3 * A) ≈ 3 && inner(3 * A, A) ≈ 3
    @test norm(2 * A + 3 * B) ≈ sqrt(13) && norm(zero(SO)) ≈ 0
end

@testset "show" begin
    @test endswith(sprint(show, zero(SO)), "(0)")
    @test !occursin("*", sprint(show, SiteOperator(A)))     # a unit coefficient is elided
    @test occursin("2.0", sprint(show, 2 * A))
    @test occursin(" + ", sprint(show, SiteOperator(A) + SiteOperator(B)))
end

@testset "instantiation" begin
    @test removeunit(OpSum.matrixunit(scalarop(2.0, sectortype(V)), V), 3) ≈ 2 * id(V)
    @test OpSum.matrixunit(2 * B, V) ≈ 2 * OpSum.matrixunit(B, V)
    @test_throws ArgumentError OpSum.matrixunit(zero(SO), V)
end
