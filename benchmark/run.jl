#!/usr/bin/env julia
#
# Benchmark driver: collects bond-dimension metrics and (optionally) timings, and writes them as
# two JSON files. Deliberately does not load CairoMakie, so plotting never perturbs the timings.
#
# Usage:
#   julia --project=benchmark benchmark/run.jl [options]
#
#   --sweep NAME      smoke | ci | full            [default: ci]
#   --models a,b      restrict to these model keys [default: all]
#   --timings PATH    BenchmarkTools results JSON  [default: benchmark/results.json]
#   --metrics PATH    bond-dimension JSON          [default: <timings stem>.metrics.json]
#   --dims-only       skip the timing suite
#   --svd             additionally time SVDBondAlgorithm
#   --internals       additionally time the intermediate term table

using BenchmarkTools
using JSON3
using Dates: now
using OpSum: OpSum

function parse_args(args)
    sweep = "ci"
    models = nothing
    timings = joinpath(@__DIR__, "results.json")
    metrics = nothing
    dims_only = false
    svd = false
    internals = false

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--sweep"
            sweep = args[i += 1]
        elseif a == "--models"
            models = String.(split(args[i += 1], ','))
        elseif a == "--timings"
            timings = args[i += 1]
        elseif a == "--metrics"
            metrics = args[i += 1]
        elseif a == "--dims-only"
            dims_only = true
        elseif a == "--svd"
            svd = true
        elseif a == "--internals"
            internals = true
        else
            error("Unknown argument: $a")
        end
        i += 1
    end

    sweep in ("smoke", "ci", "full") || error("--sweep must be smoke, ci or full (got $sweep)")
    metrics === nothing && (metrics = replace(timings, r"\.json$" => "") * ".metrics.json")
    return (; sweep, models, timings, metrics, dims_only, svd, internals)
end

const OPTS = parse_args(ARGS)

# `benchmarks.jl` reads its configuration from the environment.
ENV["OPSUM_BENCH_SWEEP"] = OPTS.sweep
OPTS.models === nothing || (ENV["OPSUM_BENCH_MODELS"] = join(OPTS.models, ','))
OPTS.svd && (ENV["OPSUM_BENCH_SVD"] = "1")
OPTS.internals && (ENV["OPSUM_BENCH_INTERNALS"] = "1")

include(joinpath(@__DIR__, "ShowcaseModels.jl"))
using .ShowcaseModels: MODELS, model_metrics

const PROVENANCE = (
    schema_version = 1,
    generated = string(now()),
    julia_version = string(VERSION),
    opsum_version = string(pkgversion(OpSum)),
    git_commit = try
        readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`)
    catch
        "unknown"
    end,
    # `git_commit` alone is misleading when the tree has uncommitted changes — the numbers then do not
    # correspond to that commit. Record it explicitly.
    git_dirty = try
        !isempty(readchomp(`git -C $(@__DIR__) status --porcelain`))
    catch
        missing
    end,
    sweep = OPTS.sweep,
    hostname = gethostname(),
)

# ── Bond-dimension metrics ────────────────────────────────────────────────────
# Deterministic and machine-independent: one `irrep_mpo` call per point. Because that is far cheaper
# than a timing measurement, these sweeps run to roughly twice the system size.

sweepsym = Symbol(OPTS.sweep)
models = Dict{String, Any}()
for spec in MODELS
    (OPTS.models === nothing || spec.key in OPTS.models) || continue
    sizes = get(spec.dimsizes, sweepsym, Int[])
    isempty(sizes) && continue
    entries = map(sizes) do N
        @info "metrics" model = spec.key N
        model_metrics(spec, N)
    end
    models[spec.key] = (
        label = spec.label,
        family = spec.family,
        params = spec.params,
        sizevar = "N",
        entries = entries,
    )
end

mkpath(dirname(abspath(OPTS.metrics)))
open(abspath(OPTS.metrics), "w") do io
    JSON3.pretty(io, merge(PROVENANCE, (; models)))
end
@info "wrote metrics" path = OPTS.metrics models = length(models)

# ── Timings ───────────────────────────────────────────────────────────────────
# Both the `include` and the `run` below are top-level statements, so each sees the world age
# created by the previous one. No `Base.invokelatest` gymnastics required.

if !OPTS.dims_only
    include(joinpath(@__DIR__, "benchmarks.jl"))
    results = run(SUITE; verbose = true)
    mkpath(dirname(abspath(OPTS.timings)))
    BenchmarkTools.save(OPTS.timings, results)
    @info "wrote timings" path = OPTS.timings
end
