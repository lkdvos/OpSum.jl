using Test
using OpSum
using OpSum: irrep_mpo, mpo_terms, instantiate
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using LinearAlgebra: norm

# The showcase models are shared with the example pages and the benchmark harness. Including them
# here means the benchmark suite can no longer silently rot against an API change: it is exercised
# by the ordinary test run. `ShowcaseModels` deliberately depends only on OpSum, TensorKit and
# LinearAlgebra, so this does not drag BenchmarkTools/JSON3/CairoMakie into the test environment.
include(joinpath(@__DIR__, "..", "benchmark", "ShowcaseModels.jl"))
using .ShowcaseModels

# Expected maximum dense-equivalent and reduced bond dimensions. All of these are independent of
# system size except the long-range models, which grow linearly.
const EXPECTED = Dict{String, Any}(
    "heisenberg_su2" => (dense = _ -> 5, mult = _ -> 3),
    "j1j2_su2" => (dense = _ -> 8, mult = _ -> 4),
    "ladder_su2" => (dense = _ -> 8, mult = _ -> 4),
    "cylinder_ly3" => (dense = _ -> 3 * 3 + 2, mult = _ -> 3 + 2),
    "cylinder_ly4" => (dense = _ -> 3 * 4 + 2, mult = _ -> 4 + 2),
    "cylinder_ly6" => (dense = _ -> 3 * 6 + 2, mult = _ -> 6 + 2),
    "alternating_spin" => (dense = _ -> 5, mult = _ -> 3),
    "free_fermions" => (dense = _ -> 4, mult = _ -> 4),
    "kitaev_chain" => (dense = _ -> 4, mult = _ -> 4),
    "tv_chain" => (dense = _ -> 5, mult = _ -> 5),
    "hubbard_1d" => (dense = _ -> 7, mult = _ -> 7),
    # One physical site per site instead of two spin-orbitals, and the spin doublet carries qdim 2,
    # so the reduced and dense-equivalent figures differ here where the abelian encoding's do not.
    "hubbard_su2" => (dense = _ -> 10, mult = _ -> 6),
    # Long-range: every pair couples, so the cover keeps ~one open spin-1 channel per site on the
    # smaller side of each cut.
    "haldane_shastry" => (dense = N -> 3 * (N ÷ 2) + 2, mult = N -> N ÷ 2 + 2),
    "powerlaw_a3" => (dense = N -> 3 * (N ÷ 2) + 2, mult = N -> N ÷ 2 + 2),
)

@testset "alphabet letter identities" begin
    # These guard against a silent reordering of TensorKit's canonical block order, which would
    # change which physical operator a given letter index denotes.
    V = Rep[U₁](0 => 1, 1 => 1)
    letters(op) = Dict(pairs(op))
    nup = matrixunit(V, U1Irrep(1), U1Irrep(1))
    @test only(pairs(letters(nup))) ==
        (IrrepOperator(U1Irrep(0), 2) => ComplexF64(1))

    Vf = Vect[FermionNumber](0 => 1, 1 => 1)
    vac, occ = FermionNumber(0), FermionNumber(1)
    @test length(letters(matrixunit(Vf, vac, occ))) == 1     # c
    @test length(letters(matrixunit(Vf, occ, vac))) == 1     # c†
    @test length(letters(matrixunit(Vf, occ, occ))) == 1     # n̂
    @test BraidingStyle(sectortype(Vf)) isa Fermionic

    # A composite on-site operator really is composite -- `couple` distributes over it.
    Sz = (matrixunit(V, U1Irrep(1), U1Irrep(1)) - matrixunit(V, U1Irrep(0), U1Irrep(0))) / 2
    @test length(letters(Sz)) == 2
end

@testset "geometry" begin
    for (Lx, Ly) in ((4, 3), (3, 4), (5, 6))
        bonds = cylinder_bonds(Lx, Ly)
        @test all(b -> b[1] < b[2], bonds)               # `couple` needs increasing site order
        @test length(bonds) == length(unique(bonds))
        @test length(bonds) == Lx * Ly + Ly * (Lx - 1)   # Lx*Ly rungs + Ly legs of length Lx-1
    end
    @test length(ladder_bonds(4, 2)) == 4 + 2 * 3
end

@testset "$(spec.key)" for spec in MODELS
    for N in spec.timesizes[:smoke]
        H, sites = spec.build(N)
        @test !isempty(H)
        @test length(sites) == N

        Ws, secs = irrep_mpo(H, sites)
        @test length(secs) == N

        # Faithfulness: the reduced MPO reconstructs every original term exactly.
        @test mpo_terms(Ws, secs) ≈ H

        if haskey(EXPECTED, spec.key)
            e = EXPECTED[spec.key]
            @test maxdensedim(secs) == e.dense(N)
            @test maxbonddim(secs) == e.mult(N)
        end

        # Hermiticity is the sharpest check on the fermionic ordering sign (`c†_j c_i = -c_i c†_j`),
        # so every fermionic model gets it. One spin model is included as a control. Restricting it
        # this way matters: `instantiate` is exponential in `N`, and running it for every model adds
        # about a minute for no extra coverage (the bosonic builders share one code path).
        if N <= 8 && (spec.family === :fermionic || spec.key == "heisenberg_su2")
            @test hermiticity_error(H, sites) < 1.0e-10
        end
    end
end

@testset "fermionic sign is required" begin
    # Flipping the h.c. sign must break hermiticity -- otherwise the check above proves nothing.
    # `couple` now supplies the sign for `couple(F.cd[i + 1], F.c[i])`, so getting it wrong takes
    # deliberately writing the site-ordered spelling with a `+`.
    F = fermion_ops()
    V = ShowcaseModels.FERMION_MODE
    N = 6
    sites = fill(V, N)
    wrong = opsum(
        couple(F.cd[i], F.c[i + 1]) + couple(F.c[i], F.cd[i + 1]) for i in 1:(N - 1)
    )
    @test hermiticity_error(wrong, sites) > 0.1
end

@testset "free-fermion spectrum" begin
    N, t = 6, 1.0
    H, sites = ShowcaseModels.free_fermions(N; t)
    ε = [-2t * cos(k * π / (N + 1)) for k in 1:N]
    exact = sort(
        [
            sum(ε[k] for k in 1:N if (m >> (k - 1)) & 1 == 1; init = 0.0) for m in 0:(2^N - 1)
        ]
    )
    @test spectrum(H, sites) ≈ exact
end

@testset "Hubbard exactly-solvable limits" begin
    Nsites = 3
    # U = 0: two independent species of free fermions.
    H0, lat0 = ShowcaseModels.hubbard(2Nsites; U = 0.0)
    ε1 = [-2 * cos(k * π / (Nsites + 1)) for k in 1:Nsites]
    ε = vcat(ε1, ε1)
    exact = sort(
        [
            sum(ε[k] for k in 1:(2Nsites) if (m >> (k - 1)) & 1 == 1; init = 0.0)
                for m in 0:(2^(2Nsites) - 1)
        ]
    )
    @test spectrum(H0, lat0) ≈ exact

    # t = 0: the energy counts doubly occupied sites in units of U.
    U = 4.0
    Ht, latt = ShowcaseModels.hubbard(2Nsites; t = 0.0, U)
    @test sort(unique(round.(spectrum(Ht, latt); digits = 8))) ≈ U .* collect(0:Nsites)
end

@testset "the two Hubbard encodings agree" begin
    # `hubbard_su2` is the only model whose bond is written out as fusion-tree matrix elements by
    # hand, so it is the only one where a wrong Clebsch-Gordan factor would go unnoticed. The check
    # that catches it: the same physics in the spin-orbital encoding, which shares no code with it —
    # `hubbard` places single-letter operators with `couple`, `hubbard_su2` `project`s a block.
    for Nsites in (2, 3)
        hsu2, latsu2 = ShowcaseModels.hubbard_su2(Nsites; t = 1.0, U = 4.0)
        horb, latorb = ShowcaseModels.hubbard(2Nsites; t = 1.0, U = 4.0)
        a, b = spectrum(hsu2, latsu2), spectrum(horb, latorb)
        @test length(a) == 4^Nsites          # the spin doublet's qdim must be unfolded
        @test a ≈ b
    end

    # ... and the t = 0 limit fixes the on-site term's normalisation independently of the hopping.
    U = 4.0
    h0, lat0 = ShowcaseModels.hubbard_su2(3; t = 0.0, U)
    @test sort(unique(round.(spectrum(h0, lat0); digits = 8))) ≈ U .* collect(0:3)
end

@testset "SU(2) and U(1) agree on the Heisenberg spectrum" begin
    # Different symmetry groups, different spaces, same operator: comparing spectra is the sharpest
    # available cross-check, and it validates the multiplet-degeneracy handling in `spectrum`.
    L = 4
    Hs, lats = ShowcaseModels.heisenberg_su2(L)

    Vu = Rep[U₁](0 => 1, 1 => 1)
    dn, up = U1Irrep(0), U1Irrep(1)
    S = spin_ops(Vu, up, dn)
    z = unit(U1Irrep)
    Hu = opsum(
        0.5 * couple(S.Sp[i], S.Sm[i + 1]; to = z) + 0.5 * couple(S.Sm[i], S.Sp[i + 1]; to = z) +
            couple(S.Sz[i], S.Sz[i + 1]; to = z) for i in 1:(L - 1)
    )
    a, b = spectrum(Hs, lats), spectrum(Hu, FiniteChain(Vu, L))
    @test length(a) == 2^L        # multiplet degeneracies must be unfolded
    @test a ≈ b
end
