"""
    ShowcaseModels

Registry of the Hamiltonians used by the OpSum.jl benchmarks, docs figures and smoke tests.

The model builders and helper utilities are shared with the example pages: this module `include`s
`examples/common.jl`, so there is exactly one definition of each model. Dependencies are limited to
`OpSum`, `TensorKit` and `LinearAlgebra` (all of which the test environment already has) so that
`test/test_showcase_models.jl` can include this file without pulling in BenchmarkTools, JSON3 or
CairoMakie. Serialization and timing live in `run.jl` / `benchmarks.jl`.
"""
module ShowcaseModels

include(joinpath(@__DIR__, "..", "examples", "common.jl"))

export ModelSpec, MODELS, model_metrics, logsizes

# Re-export the shared helpers from `examples/common.jl` so that dependents (the smoke test, the
# benchmark driver) get them from this one module.
export siteindex, cylinder_bonds, ladder_bonds
export bonddim, densedim, maxbonddim, maxdensedim, build
export mpo_matches_oracle, spectrum, hermiticity_error
# ...and OpSum's own verification helpers and operator builders, so a dependent needs one `using`.
export islossless, mpo_tensormap, spin_ops, fermion_ops, opsum, FiniteChain, OperatorSum
export spin, matrixunit, project, couple

# ── Model builders ────────────────────────────────────────────────────────────
# Each builder returns an `OperatorSum`: the terms, accumulated in one `opsum(lat, …)` pass and
# checked against the `FiniteChain` they are to be compressed over, which the container carries.

const SPIN_HALF = SU2Space(1 // 2 => 1)

function heisenberg_su2(N; J = 1.0)
    S = spin(SPIN_HALF)
    return opsum(FiniteChain(SPIN_HALF, N), J * dot(S[i], S[i + 1]) for i in 1:(N - 1))
end

function j1j2_su2(N; J1 = 1.0, J2 = 0.5)
    S = spin(SPIN_HALF)
    return opsum(
        FiniteChain(SPIN_HALF, N),
        (J1 * dot(S[i], S[i + 1]) for i in 1:(N - 1)),
        (J2 * dot(S[i], S[i + 2]) for i in 1:(N - 2)),
    )
end

const SPIN_ONE = SU2Space(1 => 1)

# The case `FiniteChain(V, N)` cannot express: a *non-uniform* lattice, one space per site. Nothing
# about the term algebra changes — only the lattice the MPO is formed over — which is the point.
function alternating_spin(N; J = 1.0)
    iseven(N) || throw(ArgumentError("alternating_spin: N=$N must be even"))
    spaces = [isodd(i) ? SPIN_HALF : SPIN_ONE for i in 1:N]
    Sa, Sb = spin(SPIN_HALF), spin(SPIN_ONE)
    op(i) = isodd(i) ? Sa : Sb
    return opsum(FiniteChain(spaces), J * dot(op(i)[i], op(i + 1)[i + 1]) for i in 1:(N - 1))
end

function haldane_shastry(N; J = 1.0)
    S = spin(SPIN_HALF)
    pref = J * π^2 / N^2
    return opsum(
        FiniteChain(SPIN_HALF, N),
        (pref / sin(π * (m - n) / N)^2) * dot(S[n], S[m])
            for n in 1:(N - 1) for m in (n + 1):N
    )
end

function powerlaw_su2(N; α = 3.0, J = 1.0)
    S = spin(SPIN_HALF)
    return opsum(
        FiniteChain(SPIN_HALF, N),
        (J * abs(m - n)^(-α)) * dot(S[n], S[m])
            for n in 1:(N - 1) for m in (n + 1):N
    )
end

function cylinder_su2(N; Ly = 4, periodic_y = true, J = 1.0)
    N % Ly == 0 || throw(ArgumentError("cylinder: N=$N must be a multiple of Ly=$Ly"))
    S = spin(SPIN_HALF)
    bonds = cylinder_bonds(div(N, Ly), Ly; periodic_y)
    return opsum(FiniteChain(SPIN_HALF, N), J * dot(S[i], S[j]) for (i, j) in bonds)
end

const FERMION_MODE = Vect[FermionNumber](0 => 1, 1 => 1)

# `couple` is order-free for abelian sectors, so the h.c. partner of `c†_i c_j` is written as it
# reads: anticommuting `c†_j c_i` into storage order costs a sign, and `couple` inserts it.
function _hopping_terms(F, bonds, t)
    return [
        -t * (couple(F.cd[i], F.c[j]) + couple(F.cd[j], F.c[i]))
            for (i, j) in bonds
    ]
end

function free_fermions(N; t = 1.0)
    F = fermion_ops()
    bonds = [(i, i + 1) for i in 1:(N - 1)]
    return opsum(FiniteChain(FERMION_MODE, N), _hopping_terms(F, bonds, t))
end

function tv_chain(N; t = 1.0, Vint = 2.0)
    F = fermion_ops()
    bonds = [(i, i + 1) for i in 1:(N - 1)]
    return opsum(
        FiniteChain(FERMION_MODE, N),
        _hopping_terms(F, bonds, t),
        (Vint * couple(F.n[i], F.n[j]) for (i, j) in bonds),
    )
end

const KITAEV_MODE = Vect[FermionParity](0 => 1, 1 => 1)

# Pairing does not conserve particle number, so this chain is graded by `FermionParity` alone — the
# symmetry is the user's choice, not the package's. `c†ᵢc†ⱼ` is charge-neutral there (1 ⊗ 1 = 0),
# which is exactly what a U(1)-graded space would refuse.
function kitaev_chain(N; t = 1.0, Δ = 0.5)
    odd, even = FermionParity(1), FermionParity(0)
    c = matrixunit(KITAEV_MODE, even, odd)
    cd = matrixunit(KITAEV_MODE, odd, even)
    bonds = [(i, i + 1) for i in 1:(N - 1)]
    return opsum(
        FiniteChain(KITAEV_MODE, N),
        (-t * (couple(cd[i], c[j]) + couple(cd[j], c[i])) for (i, j) in bonds),
        (Δ * (couple(cd[i], cd[j]) + couple(c[j], c[i])) for (i, j) in bonds),
    )
end

# Fermi-Hubbard with one spin-orbital per site, `(i, σ) -> 2(i-1) + σ`. Every operator is then a
# single alphabet letter, and the on-site `U n↑n↓` becomes an ordinary two-site term between the
# two orbitals of the same physical site. `N` counts *orbitals*, so it must be even.
orbital(i, σ) = 2 * (i - 1) + σ

function hubbard(N; t = 1.0, U = 4.0, Ly = nothing)
    iseven(N) || throw(ArgumentError("hubbard: N=$N counts spin-orbitals and must be even"))
    Nsites = div(N, 2)
    F = fermion_ops()
    sitebonds = if Ly === nothing
        [(i, i + 1) for i in 1:(Nsites - 1)]
    else
        Nsites % Ly == 0 ||
            throw(ArgumentError("hubbard: $Nsites sites must be a multiple of Ly=$Ly"))
        cylinder_bonds(div(Nsites, Ly), Ly)
    end
    hop = (
        _hopping_terms(F, [(orbital(i, σ), orbital(j, σ))], t)
            for (i, j) in sitebonds for σ in 1:2
    )
    int = (U * couple(F.n[orbital(i, 1)], F.n[orbital(i, 2)]) for i in 1:Nsites)
    return opsum(FiniteChain(FERMION_MODE, N), hop, int)
end

# Fermi-Hubbard with SU(2) spin: non-abelian *and* fermionic at once, one physical site per site
# instead of two spin-orbitals. The site space is graded by `(n, j, parity)`.
const HUBBARD_SU2 = ProductSector{Tuple{U1Irrep, SU2Irrep, FermionParity}}
const HUBBARD_SU2_SITE =
    Vect[HUBBARD_SU2]((0, 0, 0) => 1, (1, 1 // 2, 1) => 1, (2, 0, 0) => 1)

const HUB_VAC = HUBBARD_SU2((0, 0, 0))
const HUB_SINGLE = HUBBARD_SU2((1, 1 // 2, 1))
const HUB_DOUBLE = HUBBARD_SU2((2, 0, 0))

# `Σ_σ c†_{1σ} c_{2σ}` on a bond. This is the one operator in the registry that cannot be assembled
# from `matrixunit`: that needs one-dimensional sectors, and the singly-occupied sector here has
# quantum dimension 2 (it *is* the spin-½ doublet). So the bond is written as a `TensorMap` in the
# fusion-tree basis and handed to `project`, which is the general route. The four reduced elements
# are the Clebsch-Gordan factors of the doubly-occupied channel; what makes them trustworthy rather
# than merely transcribed is that `test_showcase_models.jl` checks this model's spectrum against the
# spin-orbital encoding above, which shares no code with it.
function hubbard_su2_hop()
    V = HUBBARD_SU2_SITE
    t = zeros(ComplexF64, V ⊗ V ← V ⊗ V)
    leg(a, b, c) = only(fusiontrees((a, b), c, (false, false)))
    three = HUBBARD_SU2((3, 1 // 2, 1))
    t[leg(HUB_VAC, HUB_SINGLE, HUB_SINGLE), leg(HUB_SINGLE, HUB_VAC, HUB_SINGLE)] .= 1
    t[leg(HUB_SINGLE, HUB_DOUBLE, three), leg(HUB_DOUBLE, HUB_SINGLE, three)] .= -1
    t[leg(HUB_VAC, HUB_DOUBLE, HUB_DOUBLE), leg(HUB_SINGLE, HUB_SINGLE, HUB_DOUBLE)] .= sqrt(2)
    t[leg(HUB_SINGLE, HUB_SINGLE, HUB_DOUBLE), leg(HUB_DOUBLE, HUB_VAC, HUB_DOUBLE)] .= sqrt(2)
    return t
end

function hubbard_su2(N; t = 1.0, U = 4.0)
    hop = hubbard_su2_hop()
    bond = project(-t * (hop + hop'))                                # projected once, placed per bond
    nupndn = matrixunit(HUBBARD_SU2_SITE, HUB_DOUBLE, HUB_DOUBLE)   # dim-1 sector: reachable
    return opsum(
        FiniteChain(HUBBARD_SU2_SITE, N),
        (bond[i] for i in 1:(N - 1)),
        (U * nupndn[i] for i in 1:N),
    )
end

# ── Registry ──────────────────────────────────────────────────────────────────

struct ModelSpec
    key::String
    label::String
    family::Symbol                        # :spin1d | :longrange | :quasi2d | :fermionic
    params::Dict{String, Any}
    build::Function                       # N -> OperatorSum
    timesizes::Dict{Symbol, Vector{Int}}  # :smoke | :ci | :full
    dimsizes::Dict{Symbol, Vector{Int}}
end

"Log-spaced sizes in `[lo, hi]`, snapped to multiples of `mult`."
function logsizes(lo, hi; n = 6, mult = 1)
    lo >= hi && return [mult * max(1, cld(lo, mult))]
    xs = exp.(range(log(lo), log(hi); length = n))
    return unique(sort(max.(mult, mult .* round.(Int, xs ./ mult))))
end

sweeps(smoke, ci, full) = Dict(:smoke => smoke, :ci => ci, :full => full)

# Sizes are chosen from measured costs. The exact bipartite sweep costs `Θ(M·K)` to intern the term
# suffixes plus `Θ(Σ_terms span)` for the sweep itself, where a term's *span* is the number of bonds
# between its first and last active site: each in-flight term is one edge at each bond it crosses.
# So finite-range models (spans `O(1)`, or `O(Ly)` on a cylinder) come out linear in N, while
# all-to-all models are `Θ(N³)` -- there the bond coefficient matrix `J(n,m)` restricted to
# `n <= b < m` is genuinely dense, so that is intrinsic to an exact minimum-vertex-cover sweep rather
# than an artefact. Use `SVDBondAlgorithm` with truncation for large long-range systems.
# Bond-dimension metrics cost a single `irrep_mpo` call while timings cost several samples, so the
# metric sweeps run to roughly twice the size.
#
# NOTE on reading the timing figures: with the compression linear, the plotted *total* for a
# finite-range model is now dominated by the symbolic term accumulation, not by `irrep_mpo`. The
# `phases` figure breaks the two apart, and it is the one to look at when judging the compression.
const MODELS = ModelSpec[
    ModelSpec(
        "heisenberg_su2", "Heisenberg SU(2)", :spin1d, Dict("J" => 1.0),
        heisenberg_su2,
        sweeps([8, 16], logsizes(8, 64; n = 4), logsizes(8, 2048; n = 9)),
        sweeps([8, 16], logsizes(8, 64; n = 4), logsizes(8, 4096; n = 10)),
    ),
    ModelSpec(
        "j1j2_su2", "J1-J2 SU(2)", :spin1d, Dict("J1" => 1.0, "J2" => 0.5),
        j1j2_su2,
        sweeps([8, 16], logsizes(8, 64; n = 4), logsizes(8, 2048; n = 9)),
        sweeps([8, 16], logsizes(8, 64; n = 4), logsizes(8, 4096; n = 10)),
    ),
    ModelSpec(
        "alternating_spin", "Alternating spin-1/2 / spin-1", :spin1d, Dict("J" => 1.0),
        alternating_spin,
        sweeps([4, 8], logsizes(8, 64; n = 4, mult = 2), logsizes(8, 2048; n = 9, mult = 2)),
        sweeps([4, 8], logsizes(8, 64; n = 4, mult = 2), logsizes(8, 4096; n = 10, mult = 2)),
    ),
    ModelSpec(
        "haldane_shastry", "Haldane-Shastry", :longrange, Dict("J" => 1.0),
        haldane_shastry,
        sweeps([8, 16], logsizes(8, 32; n = 3), logsizes(8, 256; n = 7)),
        sweeps([8, 16], logsizes(8, 48; n = 4), logsizes(8, 384; n = 8)),
    ),
    ModelSpec(
        "powerlaw_a3", "Power law 1/r^3", :longrange, Dict("alpha" => 3.0),
        N -> powerlaw_su2(N; α = 3.0),
        sweeps([8, 16], logsizes(8, 32; n = 3), logsizes(8, 256; n = 7)),
        sweeps([8, 16], logsizes(8, 48; n = 4), logsizes(8, 384; n = 8)),
    ),
    ModelSpec(
        "ladder_su2", "Ladder Ly=2", :quasi2d, Dict("Ly" => 2, "periodic_y" => false),
        N -> cylinder_su2(N; Ly = 2, periodic_y = false),
        sweeps([8, 16], logsizes(8, 64; n = 4, mult = 2), logsizes(8, 1024; n = 8, mult = 2)),
        sweeps([8, 16], logsizes(8, 64; n = 4, mult = 2), logsizes(8, 2048; n = 9, mult = 2)),
    ),
    ModelSpec(
        "cylinder_ly3", "Cylinder Ly=3", :quasi2d, Dict("Ly" => 3),
        N -> cylinder_su2(N; Ly = 3),
        sweeps([9, 18], logsizes(9, 63; n = 4, mult = 3), logsizes(9, 1023; n = 8, mult = 3)),
        sweeps([9, 18], logsizes(9, 63; n = 4, mult = 3), logsizes(9, 2046; n = 9, mult = 3)),
    ),
    ModelSpec(
        "cylinder_ly4", "Cylinder Ly=4", :quasi2d, Dict("Ly" => 4),
        N -> cylinder_su2(N; Ly = 4),
        sweeps([8, 16], logsizes(8, 64; n = 4, mult = 4), logsizes(8, 1024; n = 8, mult = 4)),
        sweeps([8, 16], logsizes(8, 64; n = 4, mult = 4), logsizes(8, 2048; n = 9, mult = 4)),
    ),
    ModelSpec(
        "cylinder_ly6", "Cylinder Ly=6", :quasi2d, Dict("Ly" => 6),
        N -> cylinder_su2(N; Ly = 6),
        sweeps([12, 24], logsizes(12, 60; n = 3, mult = 6), logsizes(12, 1020; n = 7, mult = 6)),
        sweeps([12, 24], logsizes(12, 60; n = 3, mult = 6), logsizes(12, 2040; n = 8, mult = 6)),
    ),
    ModelSpec(
        "free_fermions", "Free fermions", :fermionic, Dict("t" => 1.0),
        free_fermions,
        sweeps([8, 16], logsizes(8, 64; n = 4), logsizes(8, 2048; n = 9)),
        sweeps([8, 16], logsizes(8, 64; n = 4), logsizes(8, 4096; n = 10)),
    ),
    ModelSpec(
        "tv_chain", "t-V chain", :fermionic, Dict("t" => 1.0, "V" => 2.0),
        tv_chain,
        sweeps([8, 16], logsizes(8, 64; n = 4), logsizes(8, 2048; n = 9)),
        sweeps([8, 16], logsizes(8, 64; n = 4), logsizes(8, 4096; n = 10)),
    ),
    ModelSpec(
        "kitaev_chain", "Kitaev chain (parity only)", :fermionic, Dict("t" => 1.0, "Delta" => 0.5),
        kitaev_chain,
        sweeps([8, 16], logsizes(8, 64; n = 4), logsizes(8, 2048; n = 9)),
        sweeps([8, 16], logsizes(8, 64; n = 4), logsizes(8, 4096; n = 10)),
    ),
    ModelSpec(
        "hubbard_1d", "Fermi-Hubbard 1D", :fermionic, Dict("t" => 1.0, "U" => 4.0),
        hubbard,
        sweeps([8, 16], logsizes(8, 64; n = 4, mult = 2), logsizes(8, 1024; n = 8, mult = 2)),
        sweeps([8, 16], logsizes(8, 64; n = 4, mult = 2), logsizes(8, 2048; n = 9, mult = 2)),
    ),
    ModelSpec(
        "hubbard_ly4", "Fermi-Hubbard cylinder Ly=4", :fermionic, Dict("U" => 4.0, "Ly" => 4),
        N -> hubbard(N; Ly = 4),
        sweeps([16, 32], logsizes(16, 64; n = 3, mult = 8), logsizes(16, 512; n = 6, mult = 8)),
        sweeps([16, 32], logsizes(16, 64; n = 3, mult = 8), logsizes(16, 768; n = 7, mult = 8)),
    ),
    # Two sizing notes, both measured. The *sweeps* were sized when this builder still `project`ed
    # the same bond block once per bond (~2.3 s of term accumulation at N = 256); it now projects
    # once and places the `LocalOperator`, so the sizes are conservative rather than binding. The
    # *smoke* sizes are smaller still because `test_showcase_models.jl` runs `instantiate` on them:
    # this site has dimension 4 and a non-abelian fermionic sector, and N = 6 alone costs about ten
    # minutes there.
    ModelSpec(
        "hubbard_su2", "Fermi-Hubbard SU(2) spin", :fermionic, Dict("t" => 1.0, "U" => 4.0),
        hubbard_su2,
        sweeps([2, 4], logsizes(8, 64; n = 4), logsizes(8, 512; n = 7)),
        sweeps([2, 4], logsizes(8, 64; n = 4), logsizes(8, 512; n = 7)),
    ),
]

modelbykey(key) = MODELS[findfirst(m -> m.key == key, MODELS)]

"""
    model_metrics(spec, N) -> NamedTuple

Deterministic (timing-free) MPO statistics for one model at one size.
"""
function model_metrics(spec::ModelSpec, N::Int)
    H = spec.build(N)
    Ws, secs = irrep_mpo(H)
    return (
        size = N,
        nterms = length(H),
        nbonds = length(secs),
        maxdensedim = maxdensedim(secs),
        maxmultdim = maxbonddim(secs),
        densedims = [densedim(secs, b) for b in eachindex(secs)],
        algorithm = "bipartite",
    )
end

end # module
