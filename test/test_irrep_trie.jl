using Test
using OpSum
using OpSum: instantiate, fusion_resolve, irrep_trie, trie_terms, FusedTerm, ITOKey,
    passthrough, ispassthrough, bondcharges, vertexlabels, trie_key, caterpillar_trees,
    spin, scalarop, couple
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using LinearAlgebra: dot, norm, I as Id

LO(x) = OpSum.LocalOp(x)

# canonical fingerprint of a fused term for set comparison (ordering/coefficient tolerant)
function fingerprint(ft)
    ops = Tuple((o.c, o.n) for o in ft.opstring)
    bonds = Tuple(bondcharges(ft.tree))
    return (ops, bonds)
end

@testset "pass-through identity symbol" begin
    I = SU2Irrep
    pt = passthrough(I)
    @test ispassthrough(pt)
    @test pt.c == unit(I)
    @test pt.n == 0
    # enumerated letters are never pass-through
    V = SU2Space(1 // 2 => 1)
    @test all(!ispassthrough, instances(IrrepOperator, V))
    # pass-through instantiates to the structural identity id(V)
    @test convert(Array, instantiate(pt, V)) ≈ Matrix{ComplexF64}(Id, 2, 2)
end

@testset "SU(2) Sᵢ·Sⱼ fusion resolution" begin
    V = SU2Space(1 // 2 => 1)
    sites = [V, V, V]
    res = fusion_resolve(dot(spin(V)[1], spin(V)[2]), sites)

    # a singlet coupling of two spin-1 (rank-1) operators has exactly one channel
    @test length(res) == 1
    ft = only(res)

    # operator string: spin-1 letters on sites 1,2, pass-through on the idle site
    @test [o.c for o in ft.opstring] == [SU2Irrep(1), SU2Irrep(1), unit(SU2Irrep)]
    @test ispassthrough(ft.opstring[3])

    # bond charges of the caterpillar: 1 across the coupled region, 0 outside / at the total
    @test bondcharges(ft.tree) == [SU2Irrep(1), SU2Irrep(0), SU2Irrep(0)]
    @test ft.tree.coupled == unit(SU2Irrep)   # total charge is the singlet

    # reduced coefficient = ⟨½‖S‖½⟩² · (−√dim(1)) = (3/2)·(−√3)
    @test ft.coeff ≈ (3 / 2) * (-sqrt(3))

    # non-singlet target uses unit Clebsch–Gordan (no −√dim factor)
    res2 = fusion_resolve(couple(spin(V)[1], spin(V)[2]; to = SU2Irrep(1)), [V, V])
    @test only(res2).coeff ≈ (3 / 2)          # (√(3/2))² · 1
    @test bondcharges(only(res2).tree) == [SU2Irrep(1), SU2Irrep(1)]
end

@testset "caterpillar channel enumeration is bounded/canonical" begin
    I = SU2Irrep
    c = SU2Irrep(1)
    u = unit(I)
    # pairwise coupling with idle sites: unique channel
    @test length(caterpillar_trees((u, c, u, c, u), SU2Irrep(0))) == 1
    # three spin-1 to a singlet: also a single caterpillar channel (1⊗1→1→0)
    @test length(caterpillar_trees((c, c, c), SU2Irrep(0))) == 1
    # three spin-1 to a triplet: multiple intermediate channels
    @test length(caterpillar_trees((c, c, c), SU2Irrep(1))) > 1
end

@testset "SU(2) Heisenberg trie is charge-block-diagonal" begin
    V = SU2Space(1 // 2 => 1)

    # prefix sharing: S₁·S₂ and S₁·S₃ agree on site-1 operator AND bond charge → shared node
    tr = irrep_trie([dot(spin(V)[1], spin(V)[2]), dot(spin(V)[1], spin(V)[3])], [V, V, V])
    @test length(tr.children) == 1                    # single shared site-1 node
    site1_key = only(keys(tr.children))
    @test site1_key.op == IrrepOperator{SU2Irrep}(SU2Irrep(1), 1)
    @test site1_key.bond == SU2Irrep(1)
    shared = only(tr.children)
    @test length(shared.children) == 2                # diverge at site 2

    # distinct couplings on the same operator string are distinct paths (block-diagonal in bond)
    tr2 = irrep_trie(
        [
            couple(spin(V)[1], spin(V)[2]; to = SU2Irrep(0)),
            couple(spin(V)[1], spin(V)[2]; to = SU2Irrep(1)),
        ],
        [V, V],
    )
    @test length(tr2.children) == 1                   # shared site-1 (op S, bond 1)
    node = only(tr2.children)
    site2_bonds = sort!([k.bond.j for k in keys(node.children)])
    @test site2_bonds == [0 // 1, 1 // 1]             # two distinct total-charge sectors
end

@testset "round-trip: trie ↔ fused-term list" begin
    V = SU2Space(1 // 2 => 1)
    sites = [V, V, V, V]
    terms = [dot(spin(V)[i], spin(V)[i + 1]) for i in 1:3]

    resolved = reduce(vcat, fusion_resolve(t, sites) for t in terms)
    tr = irrep_trie(terms, sites)
    back = trie_terms(tr)

    @test length(back) == length(resolved)
    @test Set(fingerprint.(back)) == Set(fingerprint.(resolved))
    # coefficients round-trip (keyed by fingerprint, no collisions on this chain)
    in_by = Dict(fingerprint(ft) => ft.coeff for ft in resolved)
    for ft in back
        @test ft.coeff ≈ in_by[fingerprint(ft)]
    end
end

@testset "U(1) hopping term" begin
    V = Rep[U₁](0 => 1, 1 => 1)
    raise = LO(IrrepOperator(U1Irrep(1), 1))
    lower = LO(IrrepOperator(U1Irrep(-1), 1))

    # number-conserving hop: total charge 0, bonds +1 across the hop, 0 outside
    res = fusion_resolve(dot(raise[1], lower[2]), [V, V, V])
    ft = only(res)
    @test [o.c for o in ft.opstring] == [U1Irrep(1), U1Irrep(-1), unit(U1Irrep)]
    @test bondcharges(ft.tree) == [U1Irrep(1), U1Irrep(0), U1Irrep(0)]
    @test ft.coeff ≈ -1.0                              # abelian: dim = 1

    # coupling two +1 operators to total charge +2
    res2 = fusion_resolve(couple(raise[1], raise[2]; to = U1Irrep(2)), [V, V])
    @test bondcharges(only(res2).tree) == [U1Irrep(1), U1Irrep(2)]

    # trie is block-diagonal in U(1) charge
    tr = irrep_trie([dot(raise[1], lower[2]), dot(raise[2], lower[3])], [V, V, V])
    @test length(trie_terms(tr)) == 2
end

@testset "trivial-sector regression bridge (ℂ²)" begin
    V = ℂ^2
    ops = instances(IrrepOperator, V)
    # couple two nontrivial trivial-sector letters to the (trivial) singlet
    term = couple(LO(ops[2])[1], LO(ops[3])[2]; to = unit(Trivial))
    res = fusion_resolve(term, [V, V])
    ft = only(res)
    @test bondcharges(ft.tree) == [unit(Trivial), unit(Trivial)]
    @test ft.coeff ≈ -1.0                              # −√dim(trivial) = −1
    @test !ispassthrough(ft.opstring[1]) && !ispassthrough(ft.opstring[2])

    # a scalar (identity) field resolves to an all-pass-through string
    resid = fusion_resolve(scalarop(2.0, V)[1], [V, V])
    ft0 = only(resid)
    @test all(ispassthrough, ft0.opstring)
    @test ft0.coeff ≈ 2.0

    # round-trip on a trivial-sector chain
    tr = irrep_trie([term], [V, V])
    @test Set(fingerprint.(trie_terms(tr))) == Set(fingerprint.(res))
end
