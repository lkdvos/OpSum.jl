#!/usr/bin/env julia
# Plot OpSum.jl benchmark results: maximum bond dimension and construction time versus system size.
#
# Usage:
#   julia --project=benchmark scripts/plot_benchmarks.jl [options]
#
#   --run                Run the benchmark suite (as a subprocess) before plotting.
#   --sweep NAME         smoke | ci | full                      [default: ci]
#   --models a,b         Restrict to these model keys.
#   --results PATH       Timings JSON to load/save.             [default: benchmark/results.json]
#   --metrics PATH       Bond-dimension JSON.                   [default: <results stem>.metrics.json]
#   --output  PATH       Output image path.                     [default: benchmark/scaling.png]
#   --figure  NAME       scaling | phases | profile | all       [default: scaling]
#   --dims-only          With --run: collect metrics only, skip timings.
#
# Examples:
#   # Full sweep, then the headline figure:
#   julia --project=benchmark scripts/plot_benchmarks.jl --run --sweep full
#
#   # Re-plot previously saved results:
#   julia --project=benchmark scripts/plot_benchmarks.jl --figure all

using BenchmarkTools
using CairoMakie
using JSON3
using LaTeXStrings
using Printf
using Statistics

const BENCH_DIR = joinpath(@__DIR__, "..", "benchmark")
const RUN_SCRIPT = joinpath(BENCH_DIR, "run.jl")

# ── Arg parsing ───────────────────────────────────────────────────────────────

function parse_args(args)
    run_benchmarks = false
    sweep = "ci"
    models = nothing
    results_path = joinpath(BENCH_DIR, "results.json")
    metrics_path = nothing
    output_path = joinpath(BENCH_DIR, "scaling.png")
    figure = "scaling"
    dims_only = false

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--run"
            run_benchmarks = true
        elseif a == "--dims-only"
            dims_only = true
        elseif a == "--sweep"
            sweep = args[i += 1]
        elseif a == "--models"
            models = String.(split(args[i += 1], ','))
        elseif a == "--results"
            results_path = args[i += 1]
        elseif a == "--metrics"
            metrics_path = args[i += 1]
        elseif a == "--output"
            output_path = args[i += 1]
        elseif a == "--figure"
            figure = args[i += 1]
        else
            error("Unknown argument: $a\nRun with no args to see usage.")
        end
        i += 1
    end

    figure in ("scaling", "phases", "profile", "all") ||
        error("--figure must be scaling, phases, profile or all (got $figure)")
    metrics_path === nothing &&
        (metrics_path = replace(results_path, r"\.json$" => "") * ".metrics.json")

    return (; run_benchmarks, sweep, models, results_path, metrics_path, output_path, figure, dims_only)
end

# ── Running the suite ─────────────────────────────────────────────────────────
# Spawned as a subprocess so that timings are never measured in a process that has loaded Makie,
# and so the plot script needs no world-age gymnastics around `include`.

function run_benchmarks!(opts)
    cmd = `$(Base.julia_cmd()) --project=$(BENCH_DIR) $(RUN_SCRIPT)
        --sweep $(opts.sweep) --timings $(opts.results_path) --metrics $(opts.metrics_path)`
    opts.models === nothing || (cmd = `$cmd --models $(join(opts.models, ','))`)
    opts.dims_only && (cmd = `$cmd --dims-only`)
    println("Running: ", cmd)
    return run(cmd)
end

# ── Series extraction ─────────────────────────────────────────────────────────

"""Parse the integer after '=' in keys like "N=10"."""
parse_size(key::AbstractString) = parse(Int, match(r"\d+$", key).match)

"""
Extract `(sizes, median_times_ns, lo_err, hi_err)` from a `BenchmarkGroup` whose keys look like
`"N=k"`.
"""
function extract_series(group::BenchmarkGroup)
    rows = map(collect(group)) do (key, trial)
        med = median(trial)
        lo = med.time - Statistics.quantile(trial.times, 0.25)
        hi = Statistics.quantile(trial.times, 0.75) - med.time
        (parse_size(key), med.time, max(lo, 0.0), max(hi, 0.0))
    end
    sort!(rows; by = first)
    return ([r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows])
end

"""Same 4-tuple shape, for a deterministic metric (no error bars)."""
function extract_metric_series(entry, field::Symbol)
    es = sort(collect(entry.entries); by = e -> e.size)
    sizes = Int[e.size for e in es]
    vals = Float64[getproperty(e, field) for e in es]
    z = zeros(length(vals))
    return (sizes, vals, z, z)
end

"""Elementwise sum of two series, restricted to the sizes they share."""
function add_series(a, b)
    common = sort!(collect(intersect(Set(a[1]), Set(b[1]))))
    ia = Dict(s => i for (i, s) in enumerate(a[1]))
    ib = Dict(s => i for (i, s) in enumerate(b[1]))
    return (
        common,
        [a[2][ia[s]] + b[2][ib[s]] for s in common],
        [a[3][ia[s]] + b[3][ib[s]] for s in common],
        [a[4][ia[s]] + b[4][ib[s]] for s in common],
    )
end

"""
Fit `y = a * N^b` by ordinary least squares in log-log space.
Returns `(a, b, fitted)`; `b` is `NaN` when there are too few points to fit.
"""
function fit_power_law(sizes, vals)
    length(sizes) < 3 && return (NaN, NaN, fill(NaN, length(sizes)))
    x = log.(Float64.(sizes))
    y = log.(max.(vals, eps()))
    var(x) ≈ 0 && return (NaN, NaN, fill(NaN, length(sizes)))
    b = cov(x, y) / var(x)
    a = exp(mean(y) - b * mean(x))
    return a, b, a .* Float64.(sizes) .^ b
end

"""Convert nanoseconds to a human-scaled unit."""
function auto_unit(times_ns)
    maxval = maximum(times_ns)
    maxval < 1.0e3 && return "ns", 1.0
    maxval < 1.0e6 && return "μs", 1.0e-3
    maxval < 1.0e9 && return "ms", 1.0e-6
    return "s", 1.0e-9
end

# ── Layout ────────────────────────────────────────────────────────────────────

const ROWS = [
    ("1D and long-range", ["heisenberg_su2", "j1j2_su2", "haldane_shastry", "powerlaw_a3"]),
    ("Quasi-2D", ["ladder_su2", "cylinder_ly3", "cylinder_ly4", "cylinder_ly6"]),
    ("Fermionic", ["free_fermions", "tv_chain", "hubbard_1d", "hubbard_ly4"]),
]

const MARKER_CYCLE = [:circle, :rect, :diamond, :utriangle, :dtriangle, :cross, :star5]
const COLOR_CYCLE = Makie.wong_colors()

"""
Ticks for a log-scaled axis holding small integers (bond dimensions). Makie's default would label
such a narrow range as `10^0.75`, which is unreadable; these are plain integers spanning the data.
"""
function integer_log_ticks(vals)
    isempty(vals) && return Makie.automatic
    lo, hi = minimum(vals), maximum(vals)
    (lo <= 0 || !isfinite(lo) || !isfinite(hi)) && return Makie.automatic
    cands = if hi / lo < 1.5
        unique(round.(Int, [lo, (lo + hi) / 2, hi]))
    else
        unique(round.(Int, exp10.(range(log10(lo), log10(hi); length = 5))))
    end
    filter!(>(0), cands)
    isempty(cands) && return Makie.automatic
    return (Float64.(cands), string.(cands))
end

"`axislegend` throws when nothing on the axis carries a label; skip those panels."
function safe_legend!(ax; kwargs...)
    any(p -> haskey(p.attributes, :label) && !isempty(String(to_value(p.label))), ax.scene.plots) ||
        return nothing
    try
        axislegend(ax; kwargs...)
    catch err
        @warn "could not draw legend" err
    end
    return nothing
end

function draw_series!(ax, d, idx, label; scale = 1.0, fitvar = "N", showfit = true)
    color = COLOR_CYCLE[mod1(idx, length(COLOR_CYCLE))]
    marker = MARKER_CYCLE[mod1(idx, length(MARKER_CYCLE))]
    sizes, vals, lo, hi = d
    isempty(sizes) && return NaN
    y = vals .* scale
    scatter!(ax, Float64.(sizes), y; marker, color, markersize = 9, label)
    if any(>(0), lo) || any(>(0), hi)
        errorbars!(
            ax, Float64.(sizes), y, lo .* scale, hi .* scale;
            color = (color, 0.5), whiskerwidth = 6
        )
    end
    _, b, fit = fit_power_law(sizes, y)
    if showfit && !isnan(b)
        lines!(
            ax, Float64.(sizes), fit;
            color = (color, 0.6), linestyle = :dash,
            label = latexstring(@sprintf("\\sim %s^{%.2f}", fitvar, b))
        )
    end
    return b
end

label_of(metrics, key) = haskey(metrics.models, key) ? metrics.models[key].label : key

"""Total construction time = term-sum assembly + MPO compression."""
function total_time_series(suite, key)
    haskey(suite, key) || return nothing
    g = suite[key]
    (haskey(g, "termsum") && haskey(g, "mpo_bipartite")) || return nothing
    return add_series(extract_series(g["termsum"]), extract_series(g["mpo_bipartite"]))
end

# ── Headline figure: bond dimension | construction time ───────────────────────

function plot_scaling(metrics, suite, output_path)
    fig = Figure(; size = (980, 940))

    for (row, (rowname, keys)) in enumerate(ROWS)
        present = [k for k in keys if haskey(metrics.models, k)]
        isempty(present) && continue

        ax_d = Axis(
            fig[row, 1];
            title = row == 1 ? "Maximum bond dimension" : "",
            ylabel = "$rowname\nmax dense bond dim",
            xscale = log10, yscale = log10,
            xlabel = row == length(ROWS) ? "Number of sites N" : "",
            xticklabelsvisible = row == length(ROWS),
        )
        ax_t = Axis(
            fig[row, 2];
            title = row == 1 ? "Construction time" : "",
            xscale = log10, yscale = log10,
            xlabel = row == length(ROWS) ? "Number of sites N" : "",
            xticklabelsvisible = row == length(ROWS),
        )

        # One time unit per panel, from that panel's global maximum.
        tseries = Dict(k => total_time_series(suite, k) for k in present)
        allt = [s[2] for s in values(tseries) if s !== nothing && !isempty(s[2])]
        unit, scale = isempty(allt) ? ("s", 1.0e-9) : auto_unit(reduce(vcat, allt))
        ax_t.ylabel = "Time [$unit]"

        dseries = Dict(k => extract_metric_series(metrics.models[k], :maxdensedim) for k in present)
        alld = reduce(vcat, [s[2] for s in values(dseries) if !isempty(s[2])]; init = Float64[])
        ax_d.yticks = integer_log_ticks(alld)

        for (i, key) in enumerate(present)
            lbl = label_of(metrics, key)
            draw_series!(ax_d, dseries[key], i, lbl)
            s = tseries[key]
            s === nothing || draw_series!(ax_t, s, i, lbl; scale)
        end

        for ax in (ax_d, ax_t)
            safe_legend!(ax; position = :lt, framevisible = false, labelsize = 9, nbanks = 2)
        end
    end

    Label(
        fig[0, 1:2],
        "OpSum.jl: exact symmetry-reduced MPO construction";
        fontsize = 15, font = :bold
    )
    rowgap!(fig.layout, 6)
    colgap!(fig.layout, 12)

    save(output_path, fig)
    return println("Wrote $output_path")
end

# ── Supplementary: time split by phase ────────────────────────────────────────

function plot_phases(metrics, suite, output_path)
    phases = [("termsum", "Term-sum assembly"), ("mpo_bipartite", "MPO compression")]
    keys_all = reduce(vcat, [ks for (_, ks) in ROWS])
    present = [k for k in keys_all if haskey(suite, k)]
    isempty(present) && (println("No timing data; skipping phases figure."); return nothing)

    fig = Figure(; size = (660, 900))
    for (row, (ph, phname)) in enumerate(phases)
        ax = Axis(
            fig[row, 1];
            title = phname, xscale = log10, yscale = log10,
            xlabel = "", xticklabelsvisible = false,
        )
        series = Dict(
            k => (haskey(suite[k], ph) ? extract_series(suite[k][ph]) : nothing) for k in present
        )
        allv = [s[2] for s in values(series) if s !== nothing && !isempty(s[2])]
        unit, scale = isempty(allv) ? ("s", 1.0e-9) : auto_unit(reduce(vcat, allv))
        ax.ylabel = "Time [$unit]"
        for (i, k) in enumerate(present)
            s = series[k]
            s === nothing || draw_series!(ax, s, i, label_of(metrics, k); scale)
        end
        safe_legend!(ax; position = :lt, framevisible = false, labelsize = 8, nbanks = 2)
    end

    ax = Axis(
        fig[3, 1];
        title = "Total", xscale = log10, yscale = log10,
        xlabel = "Number of sites N", ylabel = "Time",
    )
    totals = Dict(k => total_time_series(suite, k) for k in present)
    allv = [s[2] for s in values(totals) if s !== nothing && !isempty(s[2])]
    unit, scale = isempty(allv) ? ("s", 1.0e-9) : auto_unit(reduce(vcat, allv))
    ax.ylabel = "Time [$unit]"
    for (i, k) in enumerate(present)
        s = totals[k]
        s === nothing || draw_series!(ax, s, i, label_of(metrics, k); scale)
    end
    safe_legend!(ax; position = :lt, framevisible = false, labelsize = 8, nbanks = 2)

    rowgap!(fig.layout, 8)
    save(output_path, fig)
    return println("Wrote $output_path")
end

# ── Supplementary: per-bond profile ───────────────────────────────────────────
# Free from data already collected: shows *where* along the chain the bond dimension sits, which
# distinguishes a cylinder's plateau from a long-range model's triangle.

function plot_profile(metrics, output_path)
    fig = Figure(; size = (760, 420))
    ax = Axis(
        fig[1, 1];
        xlabel = "Bond index", ylabel = L"D_\mathrm{dense}",
        title = "Bond-dimension profile (largest available system)",
    )
    reps = ["heisenberg_su2", "cylinder_ly4", "haldane_shastry", "hubbard_1d"]
    i = 0
    for key in reps
        haskey(metrics.models, key) || continue
        i += 1
        e = argmax(x -> x.size, collect(metrics.models[key].entries))
        prof = Float64.(e.densedims)
        lines!(
            ax, 1:length(prof), prof;
            color = COLOR_CYCLE[mod1(i, length(COLOR_CYCLE))], linewidth = 2,
            label = "$(label_of(metrics, key)) (N=$(e.size))"
        )
    end
    i == 0 && (println("No metric data; skipping profile figure."); return nothing)
    safe_legend!(ax; position = :rt, framevisible = false, labelsize = 9)
    save(output_path, fig)
    return println("Wrote $output_path")
end

# ── Entry point ───────────────────────────────────────────────────────────────

function main(args)
    opts = parse_args(args)
    opts.run_benchmarks && run_benchmarks!(opts)

    isfile(opts.metrics_path) || error(
        """
        Metrics file not found: $(opts.metrics_path)
        Run with --run to generate it first, or pass --metrics PATH.
        """
    )
    metrics = JSON3.read(read(opts.metrics_path, String))
    println("Loaded metrics for $(length(metrics.models)) models (sweep=$(metrics.sweep))")

    suite = if isfile(opts.results_path)
        BenchmarkTools.load(opts.results_path)[1]
    else
        @warn "No timings found at $(opts.results_path); plotting bond dimensions only."
        BenchmarkGroup()
    end

    stem, ext = splitext(opts.output_path)
    if opts.figure == "all"
        plot_scaling(metrics, suite, stem * "_scaling" * ext)
        plot_phases(metrics, suite, stem * "_phases" * ext)
        plot_profile(metrics, stem * "_profile" * ext)
    elseif opts.figure == "scaling"
        plot_scaling(metrics, suite, opts.output_path)
    elseif opts.figure == "phases"
        plot_phases(metrics, suite, opts.output_path)
    else
        plot_profile(metrics, opts.output_path)
    end
    return nothing
end

main(ARGS)
