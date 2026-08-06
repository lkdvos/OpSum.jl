using Test
using OpSum
using OpSum: project, matrixunit, instantiate, spin, scalarop, couple, irrep_mpo, opsum,
    irrep_mpo_tensors, mpo_terms, Term, Terms, TermSum, total, ops, tree, bondcharges,
    caterpillar_trees, _instantiate_basis
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using TensorKit: @tensor, insertrightunit, removeunit
using VectorInterface: inner
using LinearAlgebra: dot, norm

include(joinpath(@__DIR__, "testutils.jl"))   # LO, onlyterm, physmatrix

# The alphabet index `n` depends on TensorKit's canonical block order, so the whole point of
# `project` is not to hard-code it. These spaces exercise every way that order can be non-obvious:
# a single sector, the trivial sector, an abelian group, two sectors (so the *block* sector of a
# letter entry differs from its charge), a degeneracy > 1, and fermionic braiding.
const SPACES = (
    SU2Space(1 // 2 => 1),
    ℂ^2,
    Rep[U₁](0 => 1, 1 => 1),
    SU2Space(1 // 2 => 1, 3 // 2 => 1),
    SU2Space(1 // 2 => 2),
    Vect[FermionNumber](0 => 1, 1 => 1),
)

# every candidate basis element for `Vs`/`tot`, as `(ops, tree, E)`
function candidates(Vs, tot::I) where {I <: Sector}
    K = length(Vs)
    alphabets = [instances(IrrepOperator, V) for V in Vs]
    out = Tuple{Vector{IrrepOperator{I}}, FusionTree{I}, Any}[]
    for opstup in Iterators.product(alphabets...)
        cs = ntuple(k -> opstup[k].c, K)
        for f in caterpillar_trees(cs, tot)
            letters = collect(IrrepOperator{I}, opstup)
            push!(out, (letters, f, _instantiate_basis(letters, f, Vs)))
        end
    end
    return out
end

# `inner(E, E)` in closed form -- the normalization `project` divides by
gram(ops, tot) = dim(tot) / prod(op -> dim(op.c), ops)

# The projection is exactly `c_α = inner(E_α, h) / g`, which is only correct because the candidate
# basis is orthogonal with that closed-form diagonal, and complete for the target homspace. Both
# facts are derived in `irrepprojection.jl`; this pins them numerically, including for fermionic
# sectors, where the per-site factorization behind the derivation could in principle pick up
# braiding data.
# worst deviation of the Gram matrix from `δ_αβ · g(ops, tot)`, over the whole candidate set
function gram_error(Vs, tot)
    cands = candidates(Vs, tot)
    isempty(cands) && return (nothing, nothing, 0, 0)
    diagerr = 0.0
    offerr = 0.0
    for (a, (opsa, _, Ea)) in enumerate(cands)
        diagerr = max(diagerr, abs(inner(Ea, Ea) - gram(opsa, tot)))
        for b in (a + 1):length(cands)
            offerr = max(offerr, abs(inner(Ea, cands[b][3])))
        end
    end
    return (diagerr, offerr, length(cands), dim(space(last(first(cands)))))
end

@testset "candidate basis is orthogonal, normalized by dim(tot)/Π dim(c), and complete" begin
    for V in SPACES, K in 1:2
        Vs = fill(V, K)
        for tot in sectors(fuse(ntuple(_ -> V ⊗ V', K)...))
            diagerr, offerr, ncand, nhom = gram_error(Vs, tot)
            @testset "$V K=$K tot=$tot" begin
                @test ncand > 0
                @test ncand == nhom            # completeness: one candidate per morphism
                @test diagerr < 1.0e-12        # inner(E, E) == dim(tot) / Π dim(c_k)
                @test offerr < 1.0e-12         # orthogonality
            end
        end
    end
end

@testset "K=3 candidate basis (SU(2))" begin
    V = SU2Space(1 // 2 => 1)
    for (tot, n) in ((SU2Irrep(0), 5), (SU2Irrep(1), 9))
        diagerr, offerr, ncand, nhom = gram_error(fill(V, 3), tot)
        @test ncand == n
        @test ncand == nhom
        @test diagerr < 1.0e-12
        @test offerr < 1.0e-12
    end
end

@testset "single letters project back to themselves" begin
    for V in SPACES
        for op in instances(IrrepOperator, V)
            O = instantiate(op, V)

            t = onlyterm(project(O, (1,)))
            @test t.sites == [1]
            @test only(ops(t)) == op
            @test t.coeff ≈ 1

            # the SiteOperator form is the same expansion, unplaced
            lo = project(O, V)
            @test instantiate(lo, V) ≈ O
            @test only(keys(lo)) == op
        end
    end
end

# An oracle derived without any of the K-site machinery: `instantiate` writes `1/sqrt(dim(s))` into
# the `n`-th scalar over `blocks(O)`, and `inner` weights block `s` by `dim(s)`, so the coefficient
# of letter `n` is `sqrt(dim(s)) * (n-th flat block entry)`.
@testset "K=1 agrees with a direct blocks() read-off" begin
    for V in SPACES
        I = sectortype(V)
        for c in sectors(fuse(V ⊗ V'))
            O = randn(ComplexF64, V ← V ⊗ Vect[I](c => 1))
            expected = Dict{Int, ComplexF64}()
            n = 0
            for (s, b) in blocks(O)
                for p in eachindex(b)
                    n += 1
                    expected[n] = sqrt(dim(s)) * b[p]
                end
            end

            ts = project(O, (1,); rtol = 0)
            @test length(ts) == count(v -> !iszero(v), values(expected))
            for t in ts
                op = only(ops(t))
                @test op.c == c
                @test t.coeff ≈ expected[op.n]
            end
        end
    end
end

@testset "K=1 charge-neutral input" begin
    for V in SPACES
        # the identity is not a letter: it comes back as a sum of trivial-charge letters
        ts = project(id(V), (1,))
        @test !isempty(ts)
        @test all(t -> total(t) == unit(sectortype(V)), ts)
        @test instantiate(ts, [V]) ≈ insertrightunit(id(V))

        @test instantiate(project(2.5 * id(V), (1,)), [V]) ≈ insertrightunit(2.5 * id(V))
    end
end

# The sharpest normalization tests: every coefficient below is independently pinned elsewhere in the
# suite as the *input* to `instantiate`, so recovering it exactly closes the loop.
@testset "K=2 known reduced coefficients" begin
    V = SU2Space(1 // 2 => 1)

    # S·S  (cf. test_irrep_terms.jl)
    t = onlyterm(project(instantiate(dot(spin(V)[1], spin(V)[2]), [V, V]), [1, 2]))
    @test t.sites == [1, 2]
    @test total(t) == SU2Irrep(0)
    @test bondcharges(tree(t)) == [SU2Irrep(1), SU2Irrep(0)]
    @test t.coeff ≈ (3 / 2) * (-sqrt(3))

    # bare coupling to each total: the coefficient is 3/2 for *all* of them. A stray factor of
    # dim(tot) in the normalization would give 3/2, 9/2, 15/2 instead.
    for t in 0:2
        H = couple(spin(V)[1], spin(V)[2]; to = SU2Irrep(t))
        tk = onlyterm(project(instantiate(H, [V, V]), [1, 2]))
        @test total(tk) == SU2Irrep(t)
        @test tk.coeff ≈ 3 / 2
    end

    # U(1) hopping
    Vu = Rep[U₁](0 => 1, 1 => 1)
    raise = LO(IrrepOperator(U1Irrep(1), 1))
    lower = LO(IrrepOperator(U1Irrep(-1), 1))
    H = couple(raise[1], lower[2]; to = U1Irrep(0))
    tk = onlyterm(project(instantiate(H, [Vu, Vu]), [1, 2]))
    @test [o.c for o in ops(tk)] == [U1Irrep(1), U1Irrep(-1)]
    @test tk.coeff ≈ 1

    # charged total
    H = couple(raise[1], raise[2]; to = U1Irrep(2))
    tk = onlyterm(project(instantiate(H, [Vu, Vu]), [1, 2]))
    @test total(tk) == U1Irrep(2)
    @test tk.coeff ≈ 1
end

# For K ≥ 3 the intermediate ("inner line") charge is not fixed by (charges, total), and the
# `1⊗1→1` vertex is antisymmetric — a slip in the coupling convention flips the sign of exactly
# that channel. These tests are the reason this file exists.
@testset "K=3 channels are resolved with the right sign" begin
    V = SU2Space(1 // 2 => 1)
    S = spin(V)
    expected = sqrt(1.5)^3

    for b2 in 0:2
        H = couple(couple(S[1], S[2]; to = SU2Irrep(b2)), S[3]; to = SU2Irrep(1))
        ts = project(instantiate(H, fill(V, 3)), [1, 2, 3])
        t = onlyterm(ts)
        @test bondcharges(tree(t)) == [SU2Irrep(1), SU2Irrep(b2), SU2Irrep(1)]
        @test t.coeff ≈ expected      # including the sign
    end

    # the chirality channel
    H = couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0))
    t = onlyterm(project(instantiate(H, fill(V, 3)), [1, 2, 3]))
    @test bondcharges(tree(t)) == [SU2Irrep(1), SU2Irrep(1), SU2Irrep(0)]
    @test t.coeff ≈ expected

    # a superposition of all three tot=1 channels: each coefficient must come back on its own tree,
    # with no mixing between channels
    coeffs = (0.7, -1.3, 2.1)
    Hs = [couple(couple(S[1], S[2]; to = SU2Irrep(b2)), S[3]; to = SU2Irrep(1)) for b2 in 0:2]
    h = sum(c * instantiate(H, fill(V, 3)) for (c, H) in zip(coeffs, Hs))
    ts = project(h, [1, 2, 3])
    @test length(ts) == 3
    for (b2, c) in zip(0:2, coeffs)
        t = only(t for t in ts if bondcharges(tree(t))[2] == SU2Irrep(b2))
        @test t.coeff ≈ c * expected
    end
end

# Every other round-trip here goes through `instantiate` and would pass even if `instantiate`'s own
# coupling convention were wrong. This one anchors the K=3 sign to physics.
@testset "K=3 projection reproduces the spin chirality" begin
    V = SU2Space(1 // 2 => 1)
    S = spin(V)
    Sx = ComplexF64[0 1; 1 0] / 2
    Sy = ComplexF64[0 -im; im 0] / 2
    Sz = ComplexF64[1 0; 0 -1] / 2
    Sv = (Sx, Sy, Sz)

    eps = zeros(3, 3, 3)
    eps[1, 2, 3] = eps[2, 3, 1] = eps[3, 1, 2] = 1
    eps[1, 3, 2] = eps[3, 2, 1] = eps[2, 1, 3] = -1
    chi = zeros(ComplexF64, 8, 8)                    # χ = Σ ε_abc S₁ᵃ S₂ᵇ S₃ᶜ  (kron site order)
    for a in 1:3, b in 1:3, c in 1:3
        iszero(eps[a, b, c]) && continue
        chi += eps[a, b, c] * kron(Sv[a], kron(Sv[b], Sv[c]))
    end

    H = couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0))
    h = instantiate(H, fill(V, 3))
    back = instantiate(project(h, [1, 2, 3]), fill(V, 3))
    Hmat = physmatrix(back, 3, 2)
    α = dot(vec(chi), vec(Hmat)) / dot(vec(chi), vec(chi))
    @test abs(α) > 1.0e-6
    @test Hmat ≈ α * chi
    @test α ≈ dot(vec(chi), vec(physmatrix(h, 3, 2))) / dot(vec(chi), vec(chi))   # same sign
end

@testset "random operators round-trip exactly" begin
    su2 = SU2Space(1 // 2 => 1)
    cases = (
        ("K=1 SU(2)", [su2], nothing),
        ("K=2 SU(2)", [su2, su2], nothing),
        ("K=3 SU(2)", fill(su2, 3), nothing),
        ("K=2 trivial", [ℂ^2, ℂ^2], nothing),
        ("K=2 U(1)", fill(Rep[U₁](0 => 1, 1 => 1), 2), nothing),
        ("K=2 fermionic", fill(Vect[FermionNumber](0 => 1, 1 => 1), 2), nothing),
        ("K=2 nonuniform", [su2, SU2Space(1 // 2 => 1, 3 // 2 => 1)], nothing),
        ("K=2 degenerate", fill(SU2Space(1 // 2 => 2), 2), nothing),
        ("K=2 charged", [su2, su2], SU2Irrep(1)),
        ("K=3 charged", fill(su2, 3), SU2Irrep(1)),
    )
    for (label, Vs, tot) in cases
        @testset "$label" begin
            I = sectortype(first(Vs))
            Vp = foldl(⊗, Vs)
            h = tot === nothing ? randn(ComplexF64, Vp ← Vp) :
                randn(ComplexF64, Vp ← Vp ⊗ Vect[I](tot => 1))
            hc = tot === nothing ? insertrightunit(h) : h

            ts = project(h, 1:length(Vs))
            back = instantiate(ts, Vs)
            @test back ≈ hc

            # the basis is complete, so a random symmetric operator uses all of it ...
            @test length(ts) == dim(space(hc))
            # ... and orthogonality gives Parseval
            @test norm(hc)^2 ≈ sum(abs2(t.coeff) * gram(ops(t), total(t)) for t in ts)
        end
    end

    # real eltype input promotes to the pipeline's ComplexF64
    h = randn(Float64, su2 ⊗ su2 ← su2 ⊗ su2)
    @test instantiate(project(h, [1, 2]), [su2, su2]) ≈ insertrightunit(h)
end

# Hand-checked reference, so this does not depend on `instantiate` at all. For ℂ² there is one
# 2×2 block with dim(s) = 1, so the letters are the matrix units in column-major order:
# n = 1 → E₁₁, 2 → E₂₁, 3 → E₁₂, 4 → E₂₂.
@testset "trivial sector: hand-written matrix-unit reference" begin
    V = ℂ^2
    σX = ComplexF64[0 1; 1 0]
    σZ = ComplexF64[1 0; 0 -1]
    unit_of(M) = TensorMap(copy(M), V ← V)

    @test physmatrix(instantiate(project(unit_of(σZ), (1,)), [V]), 1, 2) ≈ σZ

    for (M, expected) in (
            (σZ, Dict((1, 1) => 1.0, (4, 4) => 1.0, (1, 4) => -1.0, (4, 1) => -1.0)),
            (σX, Dict((2, 2) => 1.0, (3, 3) => 1.0, (2, 3) => 1.0, (3, 2) => 1.0)),
        )
        ts = project(unit_of(M) ⊗ unit_of(M), [1, 2])
        @test length(ts) == length(expected)
        got = Dict((ops(t)[1].n, ops(t)[2].n) => t.coeff for t in ts)
        for (idx, c) in expected
            @test got[idx] ≈ c
        end
    end
end

@testset "matrixunit and composite couple build the XXZ chain" begin
    V = Rep[U₁](0 => 1, 1 => 1)
    up, dn = U1Irrep(1), U1Irrep(0)
    z = unit(U1Irrep)

    Sp = matrixunit(V, up, dn)
    Sm = matrixunit(V, dn, up)
    Sz = (matrixunit(V, up, up) - matrixunit(V, dn, dn)) / 2

    # a matrix unit is a single letter; Sᶻ is genuinely composite
    @test length(Sp) == 1
    @test length(Sz) == 2

    # `couple` distributes over the composite operands -- no manual expansion needed
    H = couple(Sz[1], Sz[2]; to = z) +
        (1 / 2) * couple(Sp[1], Sm[2]; to = z) +
        (1 / 2) * couple(Sm[1], Sp[2]; to = z)

    # reference in the sector-ordered basis (0 => down, 1 => up)
    szm = ComplexF64[-0.5 0; 0 0.5]
    spm = ComplexF64[0 0; 1 0]
    smm = ComplexF64[0 1; 0 0]
    ref = kron(szm, szm) + (1 / 2) * (kron(spm, smm) + kron(smm, spm))
    @test physmatrix(instantiate(H, [V, V]), 2, 2) ≈ ref

    # and the same operator recovered by projecting the dense block
    ts = project(instantiate(H, [V, V]), [1, 2])
    @test physmatrix(instantiate(ts, [V, V]), 2, 2) ≈ ref

    # pairs that cannot fuse to `to` are dropped rather than erroring ...
    @test length(couple(Sz[1], Sz[2]; to = z)) == 4
    @test length(couple(Sz[1], Sp[2]; to = U1Irrep(1))) == 2
    # ... but a total charge that nothing reaches is an error
    @test_throws ArgumentError couple(Sp[1], Sp[2]; to = z)
end

@testset "projected terms feed the MPO pipeline" begin
    V = SU2Space(1 // 2 => 1)
    d = 2
    N = 3
    sites = fill(V, N)

    hbond = instantiate(dot(spin(V)[1], spin(V)[2]), [V, V])
    for (label, h) in (("S·S", hbond), ("S·S + identity", hbond + 0.3 * insertrightunit(id(V ⊗ V))))
        @testset "$label" begin
            H = opsum(sites, (project(h, [i, i + 1]) for i in 1:(N - 1)))

            Ws, secs = irrep_mpo(H)
            @test length(secs) == N

            # faithful at the reduced level
            back = mpo_terms(Ws, secs, sites)
            @test back ≈ H

            # and the assembled tensors contract to the operator
            T = irrep_mpo_tensors(Ws, secs, sites)
            @tensor Op[o1 o2 o3 bL; bR i1 i2 i3] :=
                T[1][bL o1; i1 b1] * T[2][b1 o2; i2 b2] * T[3][b2 o3; i3 bR]
            @test physmatrix(Op, N, d) ≈ physmatrix(instantiate(H), N, d)
        end
    end

    # the projected Heisenberg chain is the Heisenberg chain
    H = opsum(sites, (project(hbond, [i, i + 1]) for i in 1:(N - 1)))
    Href = opsum(sites, (dot(spin(V)[i], spin(V)[i + 1]) for i in 1:(N - 1)))
    @test physmatrix(instantiate(H), N, d) ≈ physmatrix(instantiate(Href), N, d)
end

@testset "tolerance" begin
    V = SU2Space(1 // 2 => 1)
    scalarpart = couple(
        LO(IrrepOperator(SU2Irrep(0), 1))[1], LO(IrrepOperator(SU2Irrep(0), 1))[2];
        to = SU2Irrep(0)
    )
    h = instantiate(dot(spin(V)[1], spin(V)[2]), [V, V]) + 1.0e-6 * instantiate(scalarpart, [V, V])

    @test length(project(h, [1, 2])) == 2                  # default keeps the small term
    @test length(project(h, [1, 2]; rtol = 1.0e-4)) == 1    # truncated away
    @test length(project(h, [1, 2]; atol = 1.0e-4)) == 1

    # dropping enough weight that the reconstruction is no longer faithful is an error, not a
    # silently wrong answer
    hrand = randn(ComplexF64, ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2)
    @test_throws ArgumentError project(hrand, [1, 2]; rtol = 0.5)

    # a zero operator projects to an empty TermSum rather than throwing
    @test isempty(project(zero(h), [1, 2]))
    @test iszero(project(zero(id(V)), V))
end

@testset "input validation" begin
    V = SU2Space(1 // 2 => 1)
    h = instantiate(dot(spin(V)[1], spin(V)[2]), [V, V])

    @test_throws ArgumentError project(h, [2, 1])                  # not increasing
    @test_throws ArgumentError project(h, [1, 1])                  # not unique
    @test_throws ArgumentError project(h, [1])                     # wrong label count
    @test_throws ArgumentError project(h, [1, 2, 3])
    @test_throws ArgumentError project(randn(ComplexF64, V ← V ⊗ V ⊗ V), (1,))   # numin ∉ (K, K+1)
    @test_throws ArgumentError project(randn(ComplexF64, V ← V ⊗ Vect[SU2Irrep](SU2Irrep(1) => 2)), (1,))
    @test_throws ArgumentError project(randn(ComplexF64, V ← V ⊗ Vect[SU2Irrep](SU2Irrep(0) => 1, SU2Irrep(1) => 1)), (1,))
    @test_throws ArgumentError project(randn(ComplexF64, V ← V'), (1,))          # domain ≠ codomain
    @test_throws ArgumentError project(zeros(Float64, ℝ^2 ← ℝ^2), (1,))          # wrong space type

    # the single-site SiteOperator form
    @test_throws ArgumentError project(h, V)                                     # two output legs
    @test_throws ArgumentError project(instantiate(spin(V)[1], [V]), SU2Space(1 => 1))
end
