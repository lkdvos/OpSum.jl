# Benchmark figures

`docs/src/assets/{scaling,phases,profile}.png` are copied in by hand. To refresh:

```bash
julia --project=benchmark benchmark/plot.jl --run --sweep full --figure all
cp benchmark/scaling_scaling.png docs/src/assets/scaling.png
cp benchmark/scaling_phases.png  docs/src/assets/phases.png
cp benchmark/scaling_profile.png docs/src/assets/profile.png
```

Current figures: generated 2026-10-29, sweep `full`, Julia 1.13.3, host `worker5057` (Rusty, exclusive node), commit `7d2ebb2` (clean tree; the commit is recorded in the gitignored `benchmark/results.json` as `git_commit`).
Timings are wall-clock on a shared machine; bond dimensions are deterministic.
