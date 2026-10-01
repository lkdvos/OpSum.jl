using Test
using OpSum
using OpSum: LocalOperator, nsites, project, instantiate, couple, spin, spin_ops, fermion_ops,
    scalarop, opsum, irrep_mpo, irrep_mpo_tensors, mpo_tensormap, islossless, FiniteChain, Terms,
    nterms_raw
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using TensorKit: removeunit, numind, insertrightunit
using LinearAlgebra: dot, eigvals, norm

include(joinpath(@__DIR__, "testutils.jl"))   # LO, onlyterm, physmatrix

# `instantiate` output, as the plain `V ← V` map (the trailing trivial charge leg dropped)
function densop(h, lat)
    O = instantiate(opsum(lat, h))
    return numind(O) == 2 * numout(O) ? O : removeunit(O, numind(O))
end

# Spectrum block by block, so it is valid for fermionic sectors too (`convert(Array, t)` is not).
function blockspectrum(h, lat)
    O = densop(h, lat)
    vals = Float64[]
    for (c, b) in blocks(O)
        ev = real(eigvals(Matrix(b)))
        for _ in 1:dim(c)
            append!(vals, ev)
        end
    end
    return sort!(vals)
end

# eigenvalues of one charge block, for sector-resolved oracles
function sectorspectrum(h, lat, c)
    O = densop(h, lat)
    return sort!(real(eigvals(Matrix(block(O, c)))))
end

const Vsu2 = SU2Space(1 // 2 => 1)
const Vu1 = Rep[U₁](0 => 1, 1 => 1)
const Vf = Vect[FermionNumber](0 => 1, 1 => 1)

# the two-site bond blocks the tests project, as (space, block, directly placed oracle)
function bondcases()
    S = spin(Vsu2)
    su2 = (Vsu2, instantiate(opsum([Vsu2, Vsu2], dot(S[1], S[2]))), (i, j) -> dot(S[i], S[j]))

    (; Sp, Sm, Sz) = spin_ops(Vu1, U1Irrep(1), U1Irrep(0))
    xxz(i, j) = couple(Sp[i], Sm[j]) / 2 + couple(Sm[i], Sp[j]) / 2 + 0.7 * couple(Sz[i], Sz[j])
    u1 = (Vu1, instantiate(opsum([Vu1, Vu1], xxz(1, 2))), xxz)

    F = fermion_ops(Vf)
    hop(i, j) = -(couple(F.cd[i], F.c[j]) + couple(F.cd[j], F.c[i]))
    ferm = (Vf, instantiate(opsum([Vf, Vf], hop(1, 2))), hop)

    return (su2 = su2, u1 = u1, fermion = ferm)
end

@testset "project(h) is the unplaced form of project(h, sites)" begin
    for (label, (V, h, _)) in pairs(bondcases())
        @testset "$label" begin
            B = project(h)
            @test B isa LocalOperator{sectortype(V)}
            @test nsites(B) == 2
            @test length(B) == length(project(h, [1, 2]))
            @test all(t -> t.sites == [1, 2], B)

            for sites in ([1, 2], [3, 4], [1, 3], [2, 7])
                placed = B[sites...]
                @test placed isa Terms
                @test placed == project(h, sites)
                @test placed ≈ project(h, sites)
                @test all(t -> t.sites == sites, placed)
                # keys and coefficients are shared, only the sites change
                @test [t.keys for t in placed] == [t.keys for t in B]
                @test [t.coeff for t in placed] == [t.coeff for t in B]
            end
            # the contiguous reading
            @test B[1] == project(h, [1, 2])
            @test B[4] == project(h, [4, 5])
        end
    end

    # K = 1: both readings coincide, and the two site forms agree with the SiteOperator form
    for V in (Vsu2, Vu1, Vf)
        O = randn(ComplexF64, V ← V)
        B = project(O)
        @test nsites(B) == 1
        @test B[3] == project(O, [3])
        @test B[3] == project(O, V)[3]
    end

    # K = 3, including a charged total
    S = spin(Vsu2)
    chi = couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(1))
    h3 = instantiate(opsum(fill(Vsu2, 3), chi))
    B3 = project(h3)
    @test nsites(B3) == 3
    @test B3[2] == project(h3, [2, 3, 4])
    @test B3[1, 4, 5] == project(h3, [1, 4, 5])
    @test B3[1, 4, 5] ≈ couple(couple(S[1], S[4]; to = SU2Irrep(1)), S[5]; to = SU2Irrep(1))
end

# The claim placement rests on: relabelling the sites of a term is exact, including across a gap,
# for every symmetry. Checked densely against the directly placed operator, and — since that shares
# `instantiate` with the projection — against sector-resolved spectra that do not go through the
# term algebra at all.
@testset "gap placement is exact" begin
    for (label, (V, h, direct)) in pairs(bondcases())
        @testset "$label" begin
            B = project(h)
            lat3 = FiniteChain(V, 3)
            lat4 = FiniteChain(V, 4)

            @test B[1, 3] ≈ direct(1, 3)
            @test densop(B[1, 3], lat3) ≈ densop(direct(1, 3), lat3)
            @test densop(B[2, 4], lat4) ≈ densop(direct(2, 4), lat4)
            @test densop(B[1, 4], lat4) ≈ densop(direct(1, 4), lat4)

            # and the compressed MPO, where the gap site is a pass-through the sweep reconstructs
            for (h3, lat) in ((opsum(B[1, 3]), lat3), (opsum(B[1, 4], B[2], B[1]), lat4))
                @test islossless(opsum(lat, h3))
                mpo = irrep_mpo(opsum(lat, h3))
                @test mpo_tensormap(irrep_mpo_tensors(mpo, lat)) ≈ instantiate(opsum(lat, h3))
            end
        end
    end

    @testset "fermionic hop across a gap carries the Jordan–Wigner sign" begin
        # Three hops on a triangle, 1–2, 2–3 and 1–3, the last one placed across the gap. The
        # one-particle sector is the 3×3 hopping matrix with amplitude `τ` on every bond: its
        # spectrum is `τ·{2, -1, -1}` when the three signs agree and `τ·{-2, 1, 1}` — the frustrated
        # triangle — if the gapped bond came out with the opposite sign. That flux is gauge
        # invariant, unlike `τ` itself, which is `instantiate`'s sign convention for a two-fermion
        # coupling and is read off the contiguous two-site block rather than assumed. (Particle–hole
        # symmetry hides the flux in the full spectrum, hence the sector-resolved checks.)
        (V, h, _) = bondcases().fermion
        B = project(h)
        one = FermionNumber(1)
        two = FermionNumber(2)
        b2 = block(densop(opsum(B[1]), FiniteChain(V, 2)), one)      # [0 τ; τ 0] in some order
        τ = real(sum(b2)) / 2
        @test abs(τ) ≈ 1

        lat3 = FiniteChain(V, 3)
        triangle = opsum(B[1], B[2], B[1, 3])
        @test sectorspectrum(triangle, lat3, one) ≈ sort!(τ .* [2.0, -1.0, -1.0])
        # Two particles: the free-fermion energies are the pair sums of the one-particle ones. The
        # 1↔3 hop now happens with site 2 occupied, so this is where the string across the gap acts;
        # a missing or doubled sign there would break the free-fermion structure.
        @test sectorspectrum(triangle, lat3, two) ≈ sort!(τ .* [-2.0, 1.0, 1.0])

        # the gapped hop alone is Hermitian, with the one-particle spectrum of a single bond
        gapped = opsum(B[1, 3])
        O = densop(gapped, lat3)
        @test O ≈ O'
        @test sectorspectrum(gapped, lat3, one) ≈ [-1.0, 0.0, 1.0]

        # a four-site chain from the same block has the open-chain band, and closing it with the
        # gapped `B[1, 4]` gives the flux-free ring (π flux would give ±√2 twice instead)
        lat4 = FiniteChain(V, 4)
        chain = opsum(B[i] for i in 1:3)
        @test sectorspectrum(chain, lat4, one) ≈ [2cos(k * π / 5) for k in 4:-1:1]
        ring = opsum(chain, B[1, 4])
        @test sectorspectrum(ring, lat4, one) ≈ [-2.0, 0.0, 0.0, 2.0]
    end

    @testset "SU(2) S·S across a gap" begin
        (V, h, _) = bondcases().su2
        B = project(h)
        lat3 = FiniteChain(V, 3)
        # S₁·S₃ on three spin-½: 1/4 on the 1–3 triplet (×2 for site 2), -3/4 on the singlet (×2)
        @test blockspectrum(opsum(B[1, 3]), lat3) ≈ [-3 / 4, -3 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4]
        # the triangle is (S_tot² - 9/4)/2: 3/4 on the quartet, -3/4 on the two doublets
        triangle = opsum(B[1], B[2], B[1, 3])
        @test blockspectrum(triangle, lat3) ≈ [fill(-3 / 4, 4); fill(3 / 4, 4)]
    end

    @testset "U(1) spins across a gap" begin
        (V, h, _) = bondcases().u1
        B = project(h)
        lat3 = FiniteChain(V, 3)
        # S₁·S₃-like XXZ bond across the gap: the same block spectrum as the contiguous bond, which
        # for a site in the middle is the two-site spectrum doubled
        two = blockspectrum(opsum(B[1]), FiniteChain(V, 2))
        @test blockspectrum(opsum(B[1, 3]), lat3) ≈ sort!(repeat(two, 2))
    end
end

@testset "placement errors" begin
    (V, h, _) = bondcases().su2
    B = project(h)
    @test_throws ArgumentError B[1, 2, 3]         # wrong index count (K = 2)
    @test_throws ArgumentError B[2, 1]            # non-increasing
    @test_throws ArgumentError B[1, 1]            # not unique
    @test_throws ArgumentError B[0]               # site < 1
    @test_throws ArgumentError B[0, 2]
    @test_throws ArgumentError B[-1, 3]

    # `project(h, sites)` keeps its own errors: a single label is a count mismatch, not a
    # contiguous placement
    @test_throws ArgumentError project(h, [1])
    @test_throws ArgumentError project(h, [1, 2, 3])
    @test_throws ArgumentError project(h, [2, 1])
    @test_throws ArgumentError project(h, [1, 1])
    @test_throws ArgumentError project(h, [1.0, 2.0])

    # the constructor enforces full support on 1:K
    S = spin(Vsu2)
    @test_throws ArgumentError LocalOperator(dot(S[1], S[3]))
    @test_throws ArgumentError LocalOperator(dot(S[1], S[2]), 3)
    @test_throws ArgumentError LocalOperator(dot(S[2], S[3]), 2)
    @test_throws ArgumentError LocalOperator(Terms{SU2Irrep}())           # K unknown
    @test_throws ArgumentError LocalOperator(Terms{SU2Irrep}(), 0)
    @test nsites(LocalOperator(Terms{SU2Irrep}(), 2)) == 2
    @test nsites(LocalOperator(dot(S[1], S[2]))) == 2
end

@testset "SiteOperator conversion" begin
    for V in (Vsu2, Vu1, Vf)
        O = project(randn(ComplexF64, V ← V), V)
        B = LocalOperator(O)
        @test nsites(B) == 1
        @test B[4] == O[4]
        @test B[4] ≈ O[4]
    end
    # a pass-through (scalar) part occupies the slot and places as the K = 0 term, exactly as it
    # does from the SiteOperator (test_local_couple.jl has the rest of the pass-through story)
    @test LocalOperator(scalarop(2.0, Vsu2))[3] == scalarop(2.0, Vsu2)[3]
    @test LocalOperator(spin(Vsu2) + scalarop(1.0, Vsu2))[3] == (spin(Vsu2) + scalarop(1.0, Vsu2))[3]
end

@testset "arithmetic mirrors Terms" begin
    (V, h, direct) = bondcases().su2
    B = project(h)
    S = spin(V)
    I = SU2Irrep

    # scalars, both sides, and division
    @test (2 * B)[1] ≈ 2 * B[1]
    @test (B * 2)[1] ≈ 2 * B[1]
    @test (B / 2)[1] ≈ B[1] / 2
    @test (-B)[1] ≈ -B[1]
    @test nsites(2 * B) == 2

    # + and - with equal K
    C = project(instantiate(opsum([V, V], couple(S[1], S[2]; to = I(1)))))
    @test (B + C)[1] ≈ B[1] + C[1]
    @test (B - C)[1] ≈ B[1] - C[1]
    @test length(B + C) == 2
    @test B + B ≈ 2 * B
    @test iszero(B - B)
    @test isempty(B - B)
    @test length(B - B) == 0

    # mismatched K is an error, not a silent overlap
    B3 = project(instantiate(opsum(fill(V, 3), couple(S[1], S[2], S[3]))))
    @test_throws ArgumentError B + B3
    @test_throws ArgumentError B - B3

    # zero keeps K and is the additive identity
    Z = zero(B)
    @test nsites(Z) == 2
    @test isempty(Z)
    @test iszero(Z)
    @test B + Z ≈ B
    @test isempty(Z[1])
    @test_throws ArgumentError B3 + Z

    # == and ≈
    @test B == project(h)
    @test B ≈ project(h)
    @test B != C
    @test !(B ≈ C)
    @test !(B ≈ zero(B))
    @test B != B3

    # the slot operator is a fixed point of project ∘ instantiate on its own K sites
    @test project(instantiate(opsum(fill(V, 2), B[1]))) ≈ B

    # show mentions the type and the slot count
    str = sprint(show, B)
    @test occursin("LocalOperator", str)
    @test occursin("nsites=2", str)
    @test occursin("LocalOperator", sprint(show, zero(B)))
end

@testset "copy is independent" begin
    (V, h, _) = bondcases().u1
    B = project(h)
    C = copy(B)
    @test C == B
    @test C.terms.terms !== B.terms.terms
    @test nsites(C) == nsites(B)
    # appending to the copy's bag leaves the original alone
    nraw = nterms_raw(B.terms)
    push!(C.terms.terms, first(B.terms.terms))
    @test nterms_raw(C.terms) == nraw + 1
    @test nterms_raw(B.terms) == nraw
    @test length(C) == length(B)                  # the coincident term is summed, not added
    @test B == project(h)
    @test !(C ≈ B)
end

@testset "a placed LocalOperator builds a lossless chain" begin
    N = 6
    for (label, (V, h, direct)) in pairs(bondcases())
        @testset "$label" begin
            B = project(h)
            lat = FiniteChain(V, N)
            H = opsum(B[i] for i in 1:(N - 1))
            Href = opsum(direct(i, i + 1) for i in 1:(N - 1))
            @test H ≈ Href
            @test islossless(opsum(lat, H))
            mpo = irrep_mpo(opsum(lat, H))
            @test length(mpo.bondsectors) == N
            @test mpo_tensormap(irrep_mpo_tensors(mpo, lat)) ≈ instantiate(opsum(lat, H))

            # next-nearest neighbours through the gapped placement
            Hnnn = opsum((B[i] for i in 1:(N - 1)), (0.4 * B[i, i + 2] for i in 1:(N - 2)))
            Hnnn_ref = opsum(
                (direct(i, i + 1) for i in 1:(N - 1)), (0.4 * direct(i, i + 2) for i in 1:(N - 2))
            )
            @test Hnnn ≈ Hnnn_ref
            @test islossless(opsum(lat, Hnnn))
        end
    end
end
