using Test
using OpSum
using OpSum: SiteOperator, Letter, scalarop, letter_product, site_product, written_block,
    Factor, ispassthrough
using TensorKit
using LinearAlgebra: norm
using Random: MersenneTwister

include(joinpath(@__DIR__, "..", "testutils.jl"))

# The structure constants of a basis (the product of operators on one site; the fermion sign of a product of
# operators on a chain is pinned against dense oracles in timeevolution/test_kronpower.jl): `e_x · e_y` with the charge legs fused to the unique channel
# `x.c ⊗ y.c`, from the memoised table. The oracle is the dense written-order block and dense products.

const SU2 = SU2Irrep

const Vf = Vect[FermionNumber](0 => 1, 1 => 1)
const Bf = unitbasis(Vf)
const F = fermionmode(Bf, FermionNumber(0), FermionNumber(1))

abelian_bases = [
    "spin 1/2 Trivial" => spin_basis(Trivial; spin = 1 // 2),
    "spin 1 U1" => spin_basis(U1Irrep; spin = 1),
    "fermion Trivial" => fermion_basis(),
    "fermion U1" => fermion_basis(U1Irrep),
    "fermion FermionNumber (unit basis)" => Bf,
    "hubbard Trivial x Trivial" => hubbard_basis(),
    "hubbard U1 x U1" => hubbard_basis(U1Irrep, U1Irrep),
]

@testset "letter products: $name" for (name, B) in abelian_bases
    I = sectortype(B)
    V = TensorKit.space(B)
    @testset "all pairs agree with the written-order block" begin
        for x in B.letters, y in B.letters
            P = letter_product(B, x, y)
            tot = only(x.c ⊗ y.c)
            keys = Factor{I}[Factor{I}(x, x.c), Factor{I}(y, tot)]
            blk = written_block(keys, [1, 1], [B])
            sgn = (OpSum.isfermionodd(x.c) && OpSum.isfermionodd(y.c)) ? -1 : 1
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
            Z = site_product(B, X, Y)
            @test plain(instantiate(Z, B)) ≈ plain(instantiate(X, B)) * plain(instantiate(Y, B))
        end
    end
end

@testset "fermion convention matches a[i] * b[i]" begin
    lat = FiniteChain(Bf, 1)
    dense(x) = densify(opsum(lat, x))
    cdc = site_product(Bf, F.cd, F.c)
    @test cdc ≈ F.n                                                  # c†c = n
    @test plain(instantiate(cdc, Bf)) ≈ dense(F.cd[1] * F.c[1])
    cc = site_product(Bf, F.c, F.cd)                                         # cc† = 1 - n
    @test plain(instantiate(cc, Bf)) ≈ dense(F.c[1] * F.cd[1])
    @test isempty(site_product(Bf, F.cd, F.cd))                              # c†c† = 0
    @test site_product(Bf, F.n, F.n) ≈ F.n
end

@testset "table hits and errors" begin
    B = Bf
    X = F.n + scalarop(2.0, B)
    site_product(B, X, X)
    table = B.products                                               # owned by the basis
    n = length(table)
    @test n > 0
    site_product(B, X, X)
    @test length(table) == n
    @test site_product(B, one(X), X) == X
    @test site_product(B, X, one(X)) == X

    S = spin_basis(SU2)
    @test_throws ArgumentError site_product(S, SiteOperator(S.letters[2]), SiteOperator(S.letters[2]))
end
