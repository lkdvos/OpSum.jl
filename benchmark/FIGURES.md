# Benchmark figures

The three figures the docs embed — `docs/src/assets/scaling.png`, `phases.png` and `profile.png` — are
generated artifacts, copied into place by hand: nothing in `docs/make.jl` or
`scripts/plot_benchmarks.jl` writes into `docs/src/assets/`. To refresh them:

```bash
julia --project=benchmark scripts/plot_benchmarks.jl --run --sweep full --figure all
cp benchmark/scaling_scaling.png docs/src/assets/scaling.png
cp benchmark/scaling_phases.png  docs/src/assets/phases.png
cp benchmark/scaling_profile.png docs/src/assets/profile.png
```

Then update the provenance below from the `generated`/`julia_version`/`git_commit`/`hostname` fields
of the (gitignored) `benchmark/results.json` and `benchmark/results.metrics.json`.

## Provenance of the current figures

| | |
|---|---|
| generated | 2026-07-27 |
| sweep | `full` |
| Julia | 1.12.6 |
| source | the code commit this branch's figures accompany, with a clean working tree |
| host | `ccqlin038.flatironinstitute.org` (Flatiron CCQ workstation) |

The exact commit is recorded in the regenerated JSON (`git_commit`, plus `git_dirty` — trust the numbers
only when that is `false`); it is deliberately not pinned here, since a rebase or an amended message
would silently invalidate it while the figures stayed correct.

Timings are wall-clock on a shared machine, so treat the absolute numbers as indicative and the fitted
exponents as good to about one decimal: repeating the sweep moves the finite-range compression fits
within `1.0 … 1.14` and the long-range ones over `1.9 … 2.24`. Bond dimensions are deterministic and
machine-independent.

## What the figures show

- **`scaling.png`** — maximum bond dimension and *total* construction time versus `N`, one row per
  model family. Bond dimension is flat to `N = 4096` for every finite-range model and follows
  `3(N÷2)+2` for the long-range ones.
- **`phases.png`** — the same timings split into symbolic term-sum assembly and MPO compression. The
  compression is `~N^1.0` for finite-range models and `~N^2.3` for all-to-all ones (asymptotically
  `Θ(N³)`, which is intrinsic — see `research/persistent-graph-mpo.md` §2.3); assembly dominates the
  total, at `~N^1.7`.
- **`profile.png`** — where along the chain the bond dimension sits, normalised to `bond / N` and
  log-scaled: a flat plateau for finite-range and quasi-2D models, a `min(b, N-b)` arch for
  long-range ones.
