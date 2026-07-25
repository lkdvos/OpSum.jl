# Benchmark suite for OpSum.jl.
#
# Defines `SUITE` and nothing else, so this file stays usable with PkgBenchmark / BenchmarkCI.
# Configuration is read from the environment, which lets `run.jl` drive it without redefining
# anything:
#
#   OPSUM_BENCH_SWEEP      smoke | ci | full        (default: ci)
#   OPSUM_BENCH_MODELS     comma-separated model keys (default: all)
#   OPSUM_BENCH_SVD        1 to also time SVDBondAlgorithm
#   OPSUM_BENCH_INTERNALS  1 to also time the intermediate term table

using BenchmarkTools
using OpSum
using OpSum: irrep_mpo, BipartiteAlgorithm, SVDBondAlgorithm

# `run.jl` includes this file after already loading the models; including it twice would create a
# second copy of the module and a conflicting-import warning.
isdefined(@__MODULE__, :ShowcaseModels) || include(joinpath(@__DIR__, "ShowcaseModels.jl"))
using .ShowcaseModels: MODELS, ModelSpec

const SWEEP = Symbol(get(ENV, "OPSUM_BENCH_SWEEP", "ci"))
const ONLY = let s = get(ENV, "OPSUM_BENCH_MODELS", "")
    isempty(s) ? nothing : String.(split(s, ','))
end
const WITH_SVD = get(ENV, "OPSUM_BENCH_SVD", "0") == "1"
const WITH_INTERNALS = get(ENV, "OPSUM_BENCH_INTERNALS", "0") == "1"

BenchmarkTools.DEFAULT_PARAMETERS.evals = 1
BenchmarkTools.DEFAULT_PARAMETERS.gctrial = true

# Larger systems get fewer samples and a longer budget: a single evaluation can take seconds, and
# re-running it many times buys no accuracy for a log-log scaling plot.
_params(N) = N > 192 ? (samples = 2, seconds = 300) :
    N > 64 ? (samples = 3, seconds = 60) :
    (samples = 8, seconds = 15)

selected() = [spec for spec in MODELS if ONLY === nothing || spec.key in ONLY]

SUITE = BenchmarkGroup()

for spec in selected()
    sizes = get(spec.timesizes, SWEEP, Int[])
    isempty(sizes) && continue

    g = addgroup!(SUITE, spec.key)
    g_t = addgroup!(g, "termsum")
    g_m = addgroup!(g, "mpo_bipartite")
    g_s = WITH_SVD ? addgroup!(g, "mpo_svd") : nothing
    g_i = WITH_INTERNALS ? addgroup!(g, "termtable") : nothing

    for N in sizes
        p = _params(N)
        builder = spec.build

        # The Hamiltonian is rebuilt in `setup` rather than captured, so only one model is alive at
        # a time and its (untimed) construction is excluded from the compression measurement.
        g_t["N=$N"] = @benchmarkable $builder($N) samples = p.samples seconds = p.seconds evals = 1
        g_m["N=$N"] = @benchmarkable(
            irrep_mpo(H, sites, $(BipartiteAlgorithm())),
            setup = ((H, sites) = $builder($N)),
            samples = p.samples, seconds = p.seconds, evals = 1,
        )
        if WITH_SVD
            g_s["N=$N"] = @benchmarkable(
                irrep_mpo(H, sites, $(SVDBondAlgorithm())),
                setup = ((H, sites) = $builder($N)),
                samples = min(p.samples, 3), seconds = p.seconds, evals = 1,
            )
        end
        if WITH_INTERNALS
            g_i["N=$N"] = @benchmarkable(
                OpSum.ITOTermTable(H, sites),
                setup = ((H, sites) = $builder($N)),
                samples = p.samples, seconds = p.seconds, evals = 1,
            )
        end
    end
end

# Compile the pipeline once per model at its smallest size. This is much cheaper than
# `BenchmarkTools.warmup(SUITE)`, which would re-run the expensive cases.
for spec in selected()
    N0 = minimum(get(spec.timesizes, :smoke, [8]))
    H, sites = spec.build(N0)
    irrep_mpo(H, sites)
    WITH_SVD && irrep_mpo(H, sites, SVDBondAlgorithm())
end
