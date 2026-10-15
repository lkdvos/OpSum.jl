using Test
using OpSum
using OpSum: project, instantiate, scalarop, couple, mpo, opsum, FiniteChain,
    mpo_tensors, mpo_terms, Term, Terms, total, ops, tree, bondcharges,
    caterpillar_trees, _instantiate_term, _project_units, _instantiate_terms
using OpSum: Letter
using TensorKit
using TensorKit: @tensor, insertrightunit, removeunit
using VectorInterface: inner
using LinearAlgebra: dot, norm

include(joinpath(@__DIR__, "..", "testutils.jl"))   # LO, onlyterm, physmatrix, unitbasis, spinvector, ...

# The basis-free expansion in matrix units is internal (`_project_units`); the tests of its mechanics
# call it through these two helpers: expand and place on `sites`, and rebuild densely on spaces `Vs`.
pu(h, sites; kw...) = _project_units(h; kw...)[sites...]
inst(ts, Vs) = _instantiate_terms(ts, Vs)
spinvec(V) = spinvector(unitbasis(V))
# dense operator of `H` (letters in `unitbasis(V)`) on `N` sites of `V`
dense(H, V, N) = instantiate(opsum(FiniteChain(unitbasis(V), N), H))

# The alphabet index `n` depends on TensorKit's block order. These spaces exercise every way it can be
# non-obvious: single sector, trivial sector, abelian, two sectors, degeneracy > 1, fermionic braiding.
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
    alphabets = [instances(Letter, V) for V in Vs]
    out = Tuple{Vector{Letter{I}}, FusionTree{I}, Any}[]
    for opstup in Iterators.product(alphabets...)
        cs = ntuple(k -> opstup[k].c, K)
        for f in caterpillar_trees(cs, tot)
            letters = collect(Letter{I}, opstup)
            push!(out, (letters, f, _instantiate_term(letters, 1:K, f, Vs)))
        end
    end
    return out
end

# `inner(E, E)` in closed form -- the normalization `project` divides by
gram(ops, tot) = dim(tot) / prod(op -> dim(op.c), ops)

# The projection is `c_α = inner(E_α, h) / g`, correct because the candidate basis is orthogonal with
# that diagonal and complete (derived in `projection.jl`); pinned numerically here, fermionic
# sectors included. Returns the worst deviation of the Gram matrix from `δ_αβ · g(ops, tot)`.
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
        @test ncand == n == nhom
        @test max(diagerr, offerr) < 1.0e-12
    end
end

# Oracle without the K-site machinery: the coefficient of letter `n` is `sqrt(dim(s))` times the `n`-th
# flat entry over `blocks(O)`.
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

            ts = pu(O, (1,); rtol = 0)
            @test length(ts) == count(v -> !iszero(v), values(expected))
            for t in ts
                op = only(ops(t))
                @test op.c == c
                @test t.coeff ≈ expected[op.n - (c == unit(I))]   # trivial charge: 1 is the identity
            end
        end
    end
end

@testset "K=1 charge-neutral input" begin
    for V in SPACES
        # the identity is not a letter: it comes back as a sum of trivial-charge letters
        ts = pu(id(V), (1,))
        @test !isempty(ts)
        @test all(t -> total(t) == unit(sectortype(V)), ts)
        @test inst(ts, [V]) ≈ insertrightunit(id(V))

        @test inst(pu(2.5 * id(V), (1,)), [V]) ≈ insertrightunit(2.5 * id(V))
    end
end

@testset "K=2 known reduced coefficients" begin
    V = SU2Space(1 // 2 => 1)
    S = spinvec(V)

    t = onlyterm(pu(inst(dot(S[1], S[2]), [V, V]), [1, 2]))
    @test t.sites == [1, 2]
    @test total(t) == SU2Irrep(0)
    @test bondcharges(tree(t)) == [SU2Irrep(1), SU2Irrep(0)]
    @test t.coeff ≈ (3 / 2) * (-sqrt(3))

    # bare coupling to each total: 3/2 for all of them (a stray dim(tot) would give 3/2, 9/2, 15/2)
    for t in 0:2
        H = couple(S[1], S[2]; to = SU2Irrep(t))
        tk = onlyterm(pu(inst(H, [V, V]), [1, 2]))
        @test total(tk) == SU2Irrep(t)
        @test tk.coeff ≈ 3 / 2
    end

    # U(1) hopping
    Vu = Rep[U₁](0 => 1, 1 => 1)
    raise = LO(Letter(U1Irrep(1), 1))
    lower = LO(Letter(U1Irrep(-1), 1))
    H = couple(raise[1], lower[2]; to = U1Irrep(0))
    tk = onlyterm(pu(inst(H, [Vu, Vu]), [1, 2]))
    @test [o.c for o in ops(tk)] == [U1Irrep(1), U1Irrep(-1)]
    @test tk.coeff ≈ 1

    # charged total
    H = couple(raise[1], raise[2]; to = U1Irrep(2))
    tk = onlyterm(pu(inst(H, [Vu, Vu]), [1, 2]))
    @test total(tk) == U1Irrep(2)
    @test tk.coeff ≈ 1
end

# For K ≥ 3 the inner-line charge is not fixed by (charges, total), and the `1⊗1→1` vertex is
# antisymmetric: a slip in the coupling convention flips the sign of exactly that channel.
@testset "K=3 channels are resolved with the right sign" begin
    V = SU2Space(1 // 2 => 1)
    S = spinvec(V)
    expected = sqrt(1.5)^3

    for b2 in 0:2
        H = couple(couple(S[1], S[2]; to = SU2Irrep(b2)), S[3]; to = SU2Irrep(1))
        ts = pu(inst(H, fill(V, 3)), [1, 2, 3])
        t = onlyterm(ts)
        @test bondcharges(tree(t)) == [SU2Irrep(1), SU2Irrep(b2), SU2Irrep(1)]
        @test t.coeff ≈ expected      # including the sign
    end

    # the chirality channel
    H = couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0))
    t = onlyterm(pu(inst(H, fill(V, 3)), [1, 2, 3]))
    @test bondcharges(tree(t)) == [SU2Irrep(1), SU2Irrep(1), SU2Irrep(0)]
    @test t.coeff ≈ expected
end

# Anchors the K=3 sign to physics (round-trips through `instantiate` alone could not catch a wrong convention).
@testset "K=3 projection reproduces the spin chirality" begin
    V = SU2Space(1 // 2 => 1)
    S = spinvec(V)
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
    h = inst(H, fill(V, 3))
    back = inst(pu(h, [1, 2, 3]), fill(V, 3))
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

            ts = pu(h, 1:length(Vs))
            back = inst(ts, Vs)
            @test back ≈ hc

            # the basis is complete, so a random symmetric operator uses all of it ...
            @test length(ts) == dim(space(hc))
            # ... and orthogonality gives Parseval
            @test norm(hc)^2 ≈ sum(abs2(t.coeff) * gram(ops(t), total(t)) for t in ts)
        end
    end

    # real eltype input promotes to the pipeline's ComplexF64
    h = randn(Float64, su2 ⊗ su2 ← su2 ⊗ su2)
    @test inst(pu(h, [1, 2]), [su2, su2]) ≈ insertrightunit(h)
end

# Hand-checked reference independent of `instantiate`: for ℂ² the letters are the matrix units in
# column-major order after the identity at n = 1: 2 → E₁₁, 3 → E₂₁, 4 → E₁₂, 5 → E₂₂.
@testset "trivial sector: hand-written matrix-unit reference" begin
    V = ℂ^2
    σX = ComplexF64[0 1; 1 0]
    σZ = ComplexF64[1 0; 0 -1]
    unit_of(M) = TensorMap(copy(M), V ← V)

    @test physmatrix(inst(pu(unit_of(σZ), (1,)), [V]), 1, 2) ≈ σZ

    for (M, expected) in (
            (σZ, Dict((2, 2) => 1.0, (5, 5) => 1.0, (2, 5) => -1.0, (5, 2) => -1.0)),
            (σX, Dict((3, 3) => 1.0, (4, 4) => 1.0, (3, 4) => 1.0, (4, 3) => 1.0)),
        )
        ts = pu(unit_of(M) ⊗ unit_of(M), [1, 2])
        @test length(ts) == length(expected)
        got = Dict((ops(t)[1].n, ops(t)[2].n) => t.coeff for t in ts)
        for (idx, c) in expected
            @test got[idx] ≈ c
        end
    end
end

@testset "composite couple builds the spin-1 XXZ chain" begin
    V = Rep[U₁](-1 => 1, 0 => 1, 1 => 1)
    B = unitbasis(V)
    z = unit(U1Irrep)
    (; Sp, Sm, Sz) = spinladder(B, [U1Irrep(1), U1Irrep(0), U1Irrep(-1)])

    # Sᶻ carries an identity component in `unitbasis`, so it enters through the product
    H = Sz[1] * Sz[2] +
        (1 / 2) * couple(Sp[1], Sm[2]; to = z) +
        (1 / 2) * couple(Sm[1], Sp[2]; to = z)

    # reference in TensorKit's sector order for U(1), which is m = 0, 1, -1
    szm = ComplexF64[0 0 0; 0 1 0; 0 0 -1]
    spm = zeros(ComplexF64, 3, 3)
    spm[2, 1] = spm[1, 3] = sqrt(2)
    smm = collect(spm')
    ref = kron(szm, szm) + (1 / 2) * (kron(spm, smm) + kron(smm, spm))
    @test physmatrix(dense(H, V, 2), 2, 3) ≈ ref

    # pairs that cannot fuse to `to` are dropped rather than erroring ...
    A = Sp + Sm
    @test length(couple(Sp[1], Sm[2]; to = z)) == 4
    @test length(couple(A[1], A[2]; to = z)) == 8
    # ... but a total charge that nothing reaches is an error
    @test_throws ArgumentError couple(Sp[1], Sp[2]; to = z)
end

@testset "projected terms feed the MPO pipeline" begin
    V = SU2Space(1 // 2 => 1)
    S = spinvec(V)
    B = unitbasis(V)
    d = 2
    N = 3
    lat = FiniteChain(B, N)

    hbond = dense(dot(S[1], S[2]), V, 2)
    for (label, h) in (("S·S", hbond), ("S·S + identity", hbond + 0.3 * insertrightunit(id(V ⊗ V))))
        @testset "$label" begin
            hloc = project(h, B)
            H = opsum((hloc[i, i + 1] for i in 1:(N - 1)))

            Ws, secs = reduced(mpo(opsum(lat, H)))
            @test length(secs) == N

            @test mpo_terms(Ws, secs) ≈ H

            # and the assembled tensors contract to the operator
            T = mpo_tensors(Ws, secs, lat)
            @tensor Op[o1 o2 o3 bL; bR i1 i2 i3] :=
                T[1][bL o1; i1 b1] * T[2][b1 o2; i2 b2] * T[3][b2 o3; i3 bR]
            @test physmatrix(Op, N, d) ≈ physmatrix(instantiate(opsum(lat, H)), N, d)
        end
    end
end

@testset "tolerance" begin
    V = SU2Space(1 // 2 => 1)
    S = spinvec(V)
    scalarpart = couple(
        LO(Letter(SU2Irrep(0), 2))[1], LO(Letter(SU2Irrep(0), 2))[2];   # a matrix unit
        to = SU2Irrep(0)
    )
    h = inst(dot(S[1], S[2]), [V, V]) + 1.0e-6 * inst(scalarpart, [V, V])

    @test length(pu(h, [1, 2])) == 2                  # default keeps the small term
    @test length(pu(h, [1, 2]; rtol = 1.0e-4)) == 1    # truncated away
    @test length(pu(h, [1, 2]; atol = 1.0e-4)) == 1

    # dropping enough weight that the reconstruction is unfaithful is an error
    hrand = randn(ComplexF64, ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2)
    @test_throws ArgumentError pu(hrand, [1, 2]; rtol = 0.5)

    # a zero operator projects to an empty term bag rather than throwing
    @test isempty(pu(zero(h), [1, 2]))
    @test isempty(pu(zero(id(V)), (1,)))
end

@testset "input validation" begin
    V = SU2Space(1 // 2 => 1)
    B = unitbasis(V)
    S = spinvec(V)
    h = inst(dot(S[1], S[2]), [V, V])

    @test pu(h, [2, 1]) ≈ pu(h, [1, 2])                 # out of order: re-coupled, S⃗₂·S⃗₁ = S⃗₁·S⃗₂
    # a repeated site is the on-site product: S⃗·S⃗ on one spin-½ is 3/4, once resolved in a sum
    @test convert(Array, instantiate(opsum(FiniteChain(B, 1), project(h, B)[1, 1])))[:, :, 1] ≈
        0.75 * ComplexF64[1 0; 0 1]
    @test_throws ArgumentError project(h, B)[1, 2, 3]            # wrong index count
    @test_throws ArgumentError _project_units(randn(ComplexF64, V ← V ⊗ V ⊗ V))   # numin ∉ (K, K+1)
    @test_throws ArgumentError _project_units(randn(ComplexF64, V ← V ⊗ Vect[SU2Irrep](SU2Irrep(1) => 2)))
    @test_throws ArgumentError _project_units(randn(ComplexF64, V ← V ⊗ Vect[SU2Irrep](SU2Irrep(0) => 1, SU2Irrep(1) => 1)))
    @test_throws ArgumentError _project_units(randn(ComplexF64, V ← V'))          # domain ≠ codomain
    @test_throws ArgumentError _project_units(zeros(Float64, ℝ^2 ← ℝ^2))          # wrong space type
end
