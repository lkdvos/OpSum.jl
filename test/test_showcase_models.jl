using Test
using OpSum
using OpSum: mpo, mpo_terms, instantiate
using TensorKit
using LinearAlgebra: norm

# The showcase models are shared with the examples and the benchmark harness; including them keeps those
# from rotting. `ShowcaseModels` depends only on OpSum, TensorKit and LinearAlgebra.
include(joinpath(@__DIR__, "..", "benchmark", "ShowcaseModels.jl"))
using .ShowcaseModels

# Expected maximum dense-equivalent and reduced bond dimensions (size-independent except long-range models).
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
    # start, finish and the two doublets `c†σ`, `cσ` (qdim 2): 2 + 2·2 dense, 2 + 2 reduced
    "hubbard_su2" => (dense = _ -> 6, mult = _ -> 4),
    # long-range: ~one open spin-1 channel per site on the smaller side of each cut
    "haldane_shastry" => (dense = N -> 3 * (N ÷ 2) + 2, mult = N -> N ÷ 2 + 2),
    "powerlaw_a3" => (dense = N -> 3 * (N ÷ 2) + 2, mult = N -> N ÷ 2 + 2),
)

@testset "geometry" begin
    for (Lx, Ly) in ((4, 3), (3, 4), (5, 6))
        bonds = cylinder_bonds(Lx, Ly)
        @test all(b -> b[1] < b[2], bonds)
        @test length(bonds) == length(unique(bonds))
        @test length(bonds) == Lx * Ly + Ly * (Lx - 1)
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

        @test mpo_terms(Ws, secs) ≈ H.terms

        if haskey(EXPECTED, spec.key)
            e = EXPECTED[spec.key]
            @test maxdensedim(secs) == e.dense(N)
            @test maxbonddim(secs) == e.mult(N)
        end

        # hermiticity checks the fermionic ordering sign; `instantiate` is exponential, so one spin control only
        if N <= 8 && (spec.family === :fermionic || spec.key == "heisenberg_su2")
            @test hermiticity_error(H) < 1.0e-10
        end
    end
end

@testset "fermionic sign is required" begin
    # the wrong relative sign must break hermiticity, else the check above proves nothing
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

    U = 4.0
    Ht = ShowcaseModels.hubbard(2Nsites; t = 0.0, U)
    @test sort(unique(round.(spectrum(Ht); digits = 8))) ≈ U .* collect(0:Nsites)
end

@testset "the two Hubbard encodings agree" begin
    # `hubbard_su2` is non-abelian and fermionic at once; the spin-orbital `hubbard` shares no code with it
    for Nsites in (2, 3)
        hsu2 = ShowcaseModels.hubbard_su2(Nsites; t = 1.0, U = 4.0)
        horb = ShowcaseModels.hubbard(2Nsites; t = 1.0, U = 4.0)
        a, b = spectrum(hsu2), spectrum(horb)
        @test length(a) == 4^Nsites          # the spin doublet's qdim must be unfolded
        @test a ≈ b
    end

    # the t = 0 limit fixes the on-site normalisation
    U = 4.0
    h0 = ShowcaseModels.hubbard_su2(3; t = 0.0, U)
    @test sort(unique(round.(spectrum(h0); digits = 8))) ≈ U .* collect(0:3)
end

@testset "SU(2) and U(1) agree on the Heisenberg spectrum" begin
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
