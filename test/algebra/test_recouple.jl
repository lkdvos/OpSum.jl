using Test
using OpSum
using OpSum: Terms, Factor, Letter, recouple, concat
using TensorKit
using Random: MersenneTwister, randperm
using TensorKitTensors.HubbardOperators: HubbardOperators as HO

include(joinpath(@__DIR__, "..", "testutils.jl"))

# Re-coupling of written-order caterpillars (`recouple`: permuted legs, `concat`: two caterpillars fused),
# through TensorKit's fusion-tree `permute` and `merge`. The oracle is the written-order product,
# materialised by `written_block`.

dense(lat, x) = plain(instantiate(opsum(lat, x)))

const Vf = Vect[FermionParity](0 => 1, 1 => 1)
const Bf = unitbasis(Vf)
const F = fermionmode(Bf, FermionParity(0), FermionParity(1))

# the SU(2) Hubbard site, in a basis built from the spin-summed hopping pair
const P, Sy = U1Irrep, SU2Irrep
const Bh = Basis([HO.e_plus_e_min(ComplexF64, P, Sy), HO.e_min_e_plus(ComplexF64, P, Sy), HO.e_num(ComplexF64, P, Sy)])
const E = project(HO.e_plus_e_min(ComplexF64, P, Sy), Bh)
const lath = FiniteChain(Bh, 4)

# Re-coupling oracle: the written-order caterpillar materialised by `written_block`, against the site-order terms.
function recouples(sites, keys, coeff, Vs)
    I = sectortype(first(Vs))
    bag = OpSum.TermBag{I}()
    OpSum.written!(bag, sites, keys, ComplexF64(coeff))
    return OpSum.dense(Terms{I}(bag), Vs) ≈ coeff * OpSum.written_block(keys, sites, Vs)
end

@testset "non-abelian re-coupling against the written-order product" begin
    J = SU2Irrep
    B1 = unitbasis(SU2Space(1 // 2 => 1))
    S1 = spinvector(B1)
    letter(x) = only(only(x).factors).letter
    Vs = fill(B1, 4)
    for j in (0, 1, 2), tot in J(j) ⊗ J(1)                       # every channel, three sites
        a = only(couple(S1[1], S1[3]; to = J(j)))
        keys = vcat(a.factors, OpSum.Factor{J}(letter(S1[2]), tot))
        @test recouples([1, 3, 2], keys, 0.7, Vs)
        @test recouples([3, 2, 1], keys, 1.0, Vs)
    end
    a = only(couple(couple(S1[1], S1[3]; to = J(2)), S1[4]; to = J(1)))   # four sites, two orders
    keys = vcat(a.factors, OpSum.Factor{J}(letter(S1[2]), J(1)))
    @test recouples([1, 3, 4, 2], keys, 0.7, Vs)
    @test recouples([4, 1, 3, 2], keys, 1.0, Vs)
    # `couple` out of order gives the same terms
    a = only(couple(S1[1], S1[3]; to = J(1)))
    x = couple(Terms{J}([a]), S1[2]; to = J(1))
    keys = vcat(a.factors, OpSum.Factor{J}(letter(S1[2]), J(1)))
    @test OpSum.dense(x, Vs) ≈
        a.coeff * only(S1[2]).coeff * OpSum.written_block(keys, [1, 3, 2], Vs)

    # under an abelian symmetry the re-coupling is the R-symbol bookkeeping
    t = only(couple(F.cd[1], F.c[2], F.cd[3], F.c[4]))
    for p in ([3, 1, 4, 2], [2, 4, 1, 3], [4, 3, 2, 1])
        bag = OpSum.TermBag{FermionParity}()
        OpSum.written!(bag, p, t.factors, 1.0 + 0im)
        out = Terms{FermionParity}(bag)
        @test Terms{FermionParity}(OpSum.recouple(p, t.factors, 1.0 + 0im)) ≈ out
    end

    # fermions under SU(2): E[q, p] = E[p, q]†, and interleaved products of spin-summed hoppings
    @test dense(lath, E[3, 1]) ≈ dense(lath, E[1, 3])'
    D(x) = dense(lath, x)
    @test D(E[1, 3] * E[2, 4]) ≈ D(E[1, 3]) * D(E[2, 4])
end

# every tree of a re-coupling comes with an orthogonal (unitary) transformation of the tree basis, so a
# single caterpillar expands into terms whose squared coefficients sum to one, none of them rounding noise
@testset "re-coupling is unitary, and keeps no noise terms" begin
    rng = MersenneTwister(7)
    letter(c) = Letter{typeof(c)}(c, 1)
    function factors(charges, bonds)
        return Factor{typeof(first(charges))}[Factor(letter(c), b) for (c, b) in zip(charges, bonds)]
    end
    function norm2(ts)
        return sum(abs2, [t.coeff for t in ts])
    end
    J = SU2Irrep
    sectors = (J(1 // 2), J(1), J(3 // 2))
    ntested = 0
    for K in (3, 4, 5), _ in 1:25
        cs = [rand(rng, sectors) for _ in 1:K]
        tot = rand(rng, collect(⊗(cs...)))
        trees = OpSum.caterpillar_trees(Tuple(cs), tot)
        isempty(trees) && continue
        bonds = OpSum.bondcharges(rand(rng, trees))
        keys = factors(cs, bonds)
        perm = randperm(rng, K)
        issorted(perm) && continue
        ts = recouple(invperm(perm), keys, 1.0 + 0im)
        @test norm2(ts) ≈ 1
        @test all(t -> abs(t.coeff) > OpSum.RECOUPLE_ATOL, ts)
        @test all(t -> t.sites == 1:K && last(t.factors).bond == tot, ts)
        ntested += 1
    end
    @test ntested > 30
    # concat: two caterpillars fused to every channel their totals allow
    for (ca, cb) in (([J(1), J(1)], [J(1 // 2), J(1 // 2)]), ([J(1 // 2), J(1), J(1)], [J(1), J(3 // 2)]))
        for ta in unique(OpSum.bondcharges(f)[end] for f in OpSum.caterpillar_trees(Tuple(ca), J(1))),
                tb in (J(1), J(2)), tot in ta ⊗ tb
            fa = OpSum.caterpillar_trees(Tuple(ca), ta)
            fb = OpSum.caterpillar_trees(Tuple(cb), tb)
            (isempty(fa) || isempty(fb)) && continue
            ka = factors(ca, OpSum.bondcharges(first(fa)))
            kb = factors(cb, OpSum.bondcharges(first(fb)))
            out = concat(ka, kb, tot)
            @test sum(abs2, last.(out)) ≈ 1
            @test all(o -> abs(o[2]) > OpSum.RECOUPLE_ATOL && last(o[1]).bond == tot, out)
        end
    end
end
