#!/usr/bin/env julia
# Run benchmarks and/or plot results.
#
# Usage:
#   julia scripts/plot_benchmarks.jl [--run] [--results PATH] [--output PATH]
#
# Flags:
#   --run              Run the benchmark suite and save results before plotting.
#   --results PATH     Path to load/save results JSON.  [default: benchmark_results.json]
#   --output  PATH     Path for the output plot file.   [default: benchmark_results.pdf]
#
# Examples:
#   # Run benchmarks, save JSON, and produce a PDF:
#   julia scripts/plot_benchmarks.jl --run
#
#   # Plot previously saved results:
#   julia scripts/plot_benchmarks.jl --results my_results.json --output my_plot.pdf
#
#   # Run and save to a custom location:
#   julia scripts/plot_benchmarks.jl --run --results v2.json --output v2.pdf

using BenchmarkTools
using CairoMakie
using LaTeXStrings
using Printf
using Statistics

const BENCHMARK_FILE = joinpath(@__DIR__, "..", "benchmark", "benchmarks.jl")

# ── Arg parsing ───────────────────────────────────────────────────────────────

function parse_args(args)
    run_benchmarks = false
    results_path = "benchmark_results.json"
    output_path = "benchmark_results.png"

    i = 1
    while i <= length(args)
        if args[i] == "--run"
            run_benchmarks = true
        elseif args[i] == "--results" && i + 1 <= length(args)
            i += 1
            results_path = args[i]
        elseif args[i] == "--output" && i + 1 <= length(args)
            i += 1
            output_path = args[i]
        else
            error("Unknown argument: $(args[i])\nRun with no args to see usage.")
        end
        i += 1
    end

    if !run_benchmarks && !isfile(results_path)
        error(
            """
            Results file not found: $results_path
            Run with --run to generate it first, or pass --results PATH to an existing file.
            """
        )
    end

    return (; run_benchmarks, results_path, output_path)
end

# ── Benchmark execution ───────────────────────────────────────────────────────

function run_and_save(results_path)
    println("Loading benchmark definitions...")
    include(BENCHMARK_FILE)   # defines Main.SUITE in a newer world age

    println("Running benchmarks (this may take a while)...")
    # invokelatest is required because SUITE was defined after this function
    # was compiled, so it lives in a newer world age.
    results = Base.invokelatest() do
        run(Main.SUITE; verbose = true)
    end

    BenchmarkTools.save(results_path, results)
    println("Results saved to: $results_path")

    return results
end

# ── Helpers ───────────────────────────────────────────────────────────────────

"""Parse the integer after '=' in keys like "N=10" or "L=200"."""
parse_size(key::String) = parse(Int, match(r"\d+$", key).match)

"""
Extract (sizes, median_times_ns, lo_err_ns, hi_err_ns) from a BenchmarkGroup
whose keys are of the form "N=k" or "L=k".
"""
function extract_series(group::BenchmarkGroup)
    triples = map(collect(group)) do (key, trial)
        med = median(trial)
        lo = med.time - Statistics.quantile(trial.times, 0.25)
        hi = Statistics.quantile(trial.times, 0.75) - med.time
        (parse_size(key), med.time, max(lo, 0.0), max(hi, 0.0))
    end
    sort!(triples; by = first)
    return (
        [t[1] for t in triples],
        [t[2] for t in triples],
        [t[3] for t in triples],
        [t[4] for t in triples],
    )
end

"""
Fit a power law t = a * N^b in log-log space via OLS.
Returns (a, b, fitted_times) where fitted_times is evaluated at `sizes`.
"""
function fit_power_law(sizes, times)
    x = log.(Float64.(sizes))
    y = log.(times)
    b = cov(x, y) / var(x)
    a = exp(mean(y) - b * mean(x))
    return a, b, a .* Float64.(sizes) .^ b
end

"""Convert a nanosecond vector to (scaled_values, unit_string)."""
function auto_unit(times_ns)
    maxval = maximum(times_ns)
    maxval < 1.0e3 && return times_ns, "ns"
    maxval < 1.0e6 && return times_ns ./ 1.0e3, "μs"
    maxval < 1.0e9 && return times_ns ./ 1.0e6, "ms"
    return times_ns ./ 1.0e9, "s"
end

# ── Layout constants ──────────────────────────────────────────────────────────

const MODELS = [
    ("haldane_shastry", "Haldane-Shastry"),
    ("all_to_all_xx", "All-to-all XX"),
    ("heisenberg", "Heisenberg"),
    ("j1j2", "J₁-J₂"),
]

const MARKER_CYCLE = [:circle, :rect, :diamond, :utriangle]
const COLOR_CYCLE = Makie.wong_colors()

# ── Plotting ──────────────────────────────────────────────────────────────────

function plot_results(suite, output_path)
    # Rows: construction, bipartite optimization, SVD optimization
    phases = [
        ("construction", "Construction"),
        ("optimization_bipartite", "Optimization (Bipartite)"),
        ("optimization_svd", "Optimization (SVD)"),
    ]

    # ── Extract all data ───────────────────────────────────────────────────────
    # data[row][col] = (sizes, times_ns, lo_ns, hi_ns) or nothing if group absent
    data = [
        [
            haskey(suite[mk], ph) ? extract_series(suite[mk][ph]) : nothing
                for (mk, _) in MODELS
        ]
            for (ph, _) in phases
    ]

    # Single unit per row based on global maximum across all models
    unit_scale = Dict("ns" => 1.0, "μs" => 1.0e-3, "ms" => 1.0e-6, "s" => 1.0e-9)
    row_units = map(data) do row_data
        present = filter(!isnothing, row_data)
        isempty(present) && return "s"
        global_max = maximum(maximum(d[2]) for d in present)
        _, unit = auto_unit([global_max])
        unit
    end

    # ── Plot: 3 stacked axes, all models overlaid on each ─────────────────────
    nrows = length(phases)
    fig = Figure(; size = (600, 950))
    axs = [
        Axis(
                fig[row, 1];
                title = phases[row][2],
                xlabel = row == nrows ? "System size" : "",
                ylabel = "Time [$(row_units[row])]",
                yscale = log10,
                xscale = log10,
                yminorticksvisible = true,
                yminorgridvisible = true,
                xticklabelsvisible = row == nrows,
            )
            for row in 1:nrows
    ]

    for (col, (_, model_name)) in enumerate(MODELS)
        for (row, _) in enumerate(phases)
            d = data[row][col]
            isnothing(d) && continue

            ax = axs[row]
            scale = unit_scale[row_units[row]]

            sizes, times_ns, lo_ns, hi_ns = d
            times = times_ns .* scale
            lo = lo_ns .* scale
            hi = hi_ns .* scale

            scatter!(
                ax, Float64.(sizes), times;
                marker = MARKER_CYCLE[col],
                color = COLOR_CYCLE[col],
                markersize = 10,
                label = model_name,
            )
            errorbars!(
                ax, Float64.(sizes), times, lo, hi;
                color = (COLOR_CYCLE[col], 0.5),
                whiskerwidth = 6,
            )

            fit_a, fit_b, fit_times = fit_power_law(sizes, times)
            lines!(
                ax, Float64.(sizes), fit_times;
                color = (COLOR_CYCLE[col], 0.6),
                linestyle = :dash,
                label = latexstring(@sprintf("\\sim N^{%.2f}", fit_b)),
            )
        end
    end

    # One legend per axis: data entries first, then fit entries
    for ax in axs
        axislegend(ax; position = :lt, framevisible = false, fontsize = 9, nbanks = 2)
    end

    rowgap!(fig.layout, 8)

    save(output_path, fig)
    return println("Plot saved to: $output_path")
end

# ── Entry point ───────────────────────────────────────────────────────────────

function main(args)
    if isempty(args)
        println(
            """
            Usage: julia scripts/plot_benchmarks.jl [--run] [--results PATH] [--output PATH]

              --run              Run benchmarks before plotting (saves results to --results path).
              --results PATH     Results JSON path  [default: benchmark_results.json]
              --output  PATH     Plot output path   [default: benchmark_results.pdf]
            """
        )
        return
    end

    opts = parse_args(args)

    suite = if opts.run_benchmarks
        run_and_save(opts.results_path)
    else
        println("Loading results from: $(opts.results_path)")
        BenchmarkTools.load(opts.results_path)[1]
    end

    return plot_results(suite, opts.output_path)
end

main(ARGS)
