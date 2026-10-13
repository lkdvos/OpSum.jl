using Test
using OpSum
using OpSum: SiteOperator, IrrepOperator, scalarop, _letter_product, _mul, _written_block, _product_table,
    ITOKey, ispassthrough
using TensorKit
using LinearAlgebra: norm
using Random: MersenneTwister

include("testutils.jl")

# The structure constants of a basis: `e_x · e_y` with the charge legs fused to the unique channel
# `x.c ⊗ y.c`, from the memoised table. The oracle is the dense written-order block and dense products.

const SU2 = SU2Irrep

const Vf = Vect[FermionNumber](0 => 1, 1 => 1)
const Bf = unitbasis(Vf)
const F = fermionmode(Bf, FermionNumber(0), FermionNumber(1))

abelian_bases = [
    "spin 1/2 Trivial" => spin_basis(Trivial; spin = 1 // 2),
    "spin 1/2 U1" => spin_basis(U1Irrep; spin = 1 // 2),
    "spin 1 Trivial" => spin_basis(Trivial; spin = 1),
    "spin 1 U1" => spin_basis(U1Irrep; spin = 1),
    "fermion Trivial" => fermion_basis(),
    "fermion U1" => fermion_basis(U1Irrep),
    "fermion FermionNumber (unit basis)" => Bf,
    "hubbard Trivial x Trivial" => hubbard_basis(),
    "hubbard U1 x Trivial" => hubbard_basis(U1Irrep, Trivial),
    "hubbard Trivial x U1" => hubbard_basis(Trivial, U1Irrep),
    "hubbard U1 x U1" => hubbard_basis(U1Irrep, U1Irrep),
]

@testset "letter products: $name" for (name, B) in abelian_bases
    I = sectortype(B)
    V = TensorKit.space(B)
    @testset "all pairs agree with the written-order block" begin
        for x in B.letters, y in B.letters
            P = _letter_product(B, x, y)
            tot = only(x.c ⊗ y.c)
            keys = ITOKey{I}[ITOKey{I}(x, x.c, 1), ITOKey{I}(y, tot, 1)]
            blk = _written_block(keys, [1, 1], [B])
            sgn = (OpSum._isfermionodd(x.c) && OpSum._isfermionodd(y.c)) ? -1 : 1
            if isempty(P)
                @test norm(blk) < 1.0e-12
            else
                @test all(l -> l.c == tot, P.letters)
                @test instantiate(P, B) ≈ sgn * blk
            end
        end
    end

    @testset "random neutral operators multiply as matrices" begin
        rng = MersenneTwister(1)
        neutral = [l for l in B.letters if l.c == unit(I)]
        for _ in 1:3
            X = SiteOperator(neutral, randn(rng, ComplexF64, length(neutral)))
            Y = SiteOperator(neutral, randn(rng, ComplexF64, length(neutral)))
            Z = _mul(B, X, Y)
            @test plain(instantiate(Z, B)) ≈ plain(instantiate(X, B)) * plain(instantiate(Y, B))
        end
    end
end

@testset "fermion convention matches a[i] * b[i]" begin
    lat = FiniteChain(Bf, 1)
    dense(x) = plain(instantiate(opsum(lat, x)))
    cdc = _mul(Bf, F.cd, F.c)
    @test cdc ≈ F.n                                                  # c†c = n
    @test plain(instantiate(cdc, Bf)) ≈ dense(F.cd[1] * F.c[1])
    cc = _mul(Bf, F.c, F.cd)                                         # cc† = 1 - n
    @test plain(instantiate(cc, Bf)) ≈ dense(F.c[1] * F.cd[1])
    @test isempty(_mul(Bf, F.cd, F.cd))                              # c†c† = 0
    @test _mul(Bf, F.n, F.n) ≈ F.n
end

@testset "table hits and errors" begin
    B = Bf
    X = F.n + scalarop(2.0, B)
    _mul(B, X, X)
    table = _product_table(B)
    @test table === _product_table(B)                                # found by identity
    n = length(table)
    @test n > 0
    _mul(B, X, X)
    @test length(table) == n
    @test _mul(B, one(X), X) == X
    @test _mul(B, X, one(X)) == X

    S = spin_basis(SU2)
    @test_throws ArgumentError _mul(S, SiteOperator(S.letters[1]), SiteOperator(S.letters[1]))
end
