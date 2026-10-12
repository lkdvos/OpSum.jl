using Test
using OpSum
using OpSum: mpo, mpo_terms, instantiate
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
    # Start, finish and the two doublets `c†σ`, `cσ` (the stock basis' own letters): 2 + 2·2 dense,
    # 2 + 2 reduced. The matrix-unit alphabet needed four hopping pieces here (10 and 6).
    "hubbard_su2" => (dense = _ -> 6, mult = _ -> 4),
    # Long-range: every pair couples, so the cover keeps ~one open spin-1 channel per site on the
    # smaller side of each cut.
    "haldane_shastry" => (dense = N -> 3 * (N ÷ 2) + 2, mult = N -> N ÷ 2 + 2),
    "powerlaw_a3" => (dense = N -> 3 * (N ÷ 2) + 2, mult = N -> N ÷ 2 + 2),
)

@testset "stock-basis letters" begin
    # The builders hand out one letter per operator that the model code mentions, so a model term is
    # as many letters as sites it touches: nothing is expanded behind the user's back.
    letters(op) = Dict(pairs(op))
    F = fermion_ops(U1Irrep)
    @test length(letters(F.f_num)) == 1                  # n̂
    @test BraidingStyle(sectortype(space(fermion_basis(U1Irrep)))) isa Fermionic
    S = spin_ops(SU2Irrep).S_exchange
    @test length(S[1, 2]) == 1                              # S⃗ᵢ·S⃗ⱼ is one term, one letter per site
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
        H = spec.build(N)
        @test !isempty(H)
        @test length(H.lattice) == N

        W = mpo(H)
        Ws, secs = W.Ws, W.bondsectors
        @test length(secs) == N

        # Faithfulness: the reduced MPO reconstructs every original term exactly.
        @test mpo_terms(Ws, secs) ≈ H.terms

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
            @test hermiticity_error(H) < 1.0e-10
        end
    end
end

@testset "fermionic sign is required" begin
    # Flipping the h.c. sign must break hermiticity -- otherwise the check above proves nothing.
    # Placement supplies the sign for `c†ᵢ₊₁ cᵢ` (`F.f_plus_f_min[i + 1, i]`), so getting it wrong
    # takes deliberately writing the hopping with the wrong relative sign.
    F = fermion_ops(U1Irrep)
    N = 6
    wrong = opsum(
        FiniteChain(ShowcaseModels.FERMION_BASIS, N),
        F.f_plus_f_min[i, i + 1] - F.f_plus_f_min[i + 1, i] for i in 1:(N - 1)
    )
    @test hermiticity_error(wrong) > 0.1
end

@testset "free-fermion spectrum" begin
    N, t = 6, 1.0
    H = ShowcaseModels.free_fermions(N; t)
    ε = [-2t * cos(k * π / (N + 1)) for k in 1:N]
    exact = sort(
        [
            sum(ε[k] for k in 1:N if (m >> (k - 1)) & 1 == 1; init = 0.0) for m in 0:(2^N - 1)
        ]
    )
    @test spectrum(H) ≈ exact
end

@testset "Hubbard exactly-solvable limits" begin
    Nsites = 3
    # U = 0: two independent species of free fermions.
    H0 = ShowcaseModels.hubbard(2Nsites; U = 0.0)
    ε1 = [-2 * cos(k * π / (Nsites + 1)) for k in 1:Nsites]
    ε = vcat(ε1, ε1)
    exact = sort(
        [
            sum(ε[k] for k in 1:(2Nsites) if (m >> (k - 1)) & 1 == 1; init = 0.0)
                for m in 0:(2^(2Nsites) - 1)
        ]
    )
    @test spectrum(H0) ≈ exact

    # t = 0: the energy counts doubly occupied sites in units of U.
    U = 4.0
    Ht = ShowcaseModels.hubbard(2Nsites; t = 0.0, U)
    @test sort(unique(round.(spectrum(Ht); digits = 8))) ≈ U .* collect(0:Nsites)
end

@testset "the two Hubbard encodings agree" begin
    # `hubbard_su2` is the only model that is non-abelian and fermionic at once, so it is the one
    # where a wrong Clebsch-Gordan factor or braiding would go unnoticed. The check that catches it:
    # the same physics in the spin-orbital encoding, which shares no code with it — `hubbard` places
    # fermion operators on spin-orbital sites, `hubbard_su2` uses the stock Hubbard site.
    for Nsites in (2, 3)
        hsu2 = ShowcaseModels.hubbard_su2(Nsites; t = 1.0, U = 4.0)
        horb = ShowcaseModels.hubbard(2Nsites; t = 1.0, U = 4.0)
        a, b = spectrum(hsu2), spectrum(horb)
        @test length(a) == 4^Nsites          # the spin doublet's qdim must be unfolded
        @test a ≈ b
    end

    # ... and the t = 0 limit fixes the on-site term's normalisation independently of the hopping.
    U = 4.0
    h0 = ShowcaseModels.hubbard_su2(3; t = 0.0, U)
    @test sort(unique(round.(spectrum(h0); digits = 8))) ≈ U .* collect(0:3)
end

@testset "SU(2) and U(1) agree on the Heisenberg spectrum" begin
    # Different symmetry groups, different spaces, same operator: comparing spectra is the sharpest
    # available cross-check, and it validates the multiplet-degeneracy handling in `spectrum`.
    L = 4
    Hs = ShowcaseModels.heisenberg_su2(L)

    S = spin_ops(U1Irrep)
    Hu = opsum(
        FiniteChain(spin_basis(U1Irrep), L),
        0.5 * S.S_plus_S_min[i, i + 1] + 0.5 * S.S_min_S_plus[i, i + 1] +
            S.S_z_S_z[i, i + 1] for i in 1:(L - 1)
    )
    a, b = spectrum(Hs), spectrum(Hu)
    @test length(a) == 2^L        # multiplet degeneracies must be unfolded
    @test a ≈ b
end
