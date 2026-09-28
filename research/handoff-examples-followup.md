# Handoff: interface fixes, then the example pages

For a fresh session orchestrating this work. Read this file first, then
`research/examples-progression.md` (the operator progression and the W1–W9 watch-list).

Everything here is grounded: the progression script in `research/progression.jl` runs green today
against `main`, and the findings below came from reading its flow rather than from speculation.

---

## 0. State of the world

* `main` is at `74f1f01` — PR #32, which made operators **latticeless**: `Terms` is the compressible
  type and the lattice is an argument to `irrep_mpo(h, lat)`. `TermSum` and `lattice` are gone;
  `FiniteChain` / `InfiniteChain` / `FiniteMPO` are new. `adjoint(h, lat)` replaced `H'`.
  `research/interface-review.md` §12 is the decision record — read it before proposing any change to
  where the lattice lives, because the alternatives were considered and rejected on record.
* This branch adds only two files: `research/progression.jl` (executable reference, 25 operators)
  and this handoff. No library code.
* The example series (`examples/`, 6 pages) was adapted to the new API but **not extended**. Missing
  entirely: infinite chains, exponential decay, the MPSKit handoff, non-uniform lattices,
  parity-only and non-abelian fermions, observables, truncation-as-a-workflow.

## 1. The plan, in order

The author agreed to this sequence. **Do not reorder it** — the point of 2 before 3 is to avoid
documenting spellings that are about to change.

1. **Preserve the reference** — done on this branch (`research/progression.jl`).
2. **Act on the two interface findings** (§2 below). Small, self-contained, testable.
3. **Write the example pages** (§3), against the settled interface.

## 2. The two interface findings

### 2a. `Terms` is missing half the mutable-container protocol

`Terms` is now *the* operator type and has a public **mutating** `append!`, but:

| method | present? |
|---|---|
| `+` `-` `*` `/`, `one`, `append!` | yes |
| `copy` | **no** |
| `zero` | **no** (though `SiteOperator` has it) |
| `empty`, `similar` | no |

Found by writing the natural non-mutating idiom, which fails:

```julia
append!(copy(hop), extra_terms)     # MethodError: no method matching copy(::Terms{...})
```

This is **ergonomics, not capability** — `h + extra` does the same job and is equally efficient for
a single addition (the quadratic problem only arises when *folding* `+`). But it is a first-hour
stumble, and `one(h)` working while `zero(h)` does not is plainly inconsistent.

Suggested scope: add `Base.copy`, `Base.zero` (both `Terms{I}` → a fresh bag), and consider
`Base.empty`. Note `copy` must copy the **vector**, not alias it — the whole point is that
`append!` and `canonicalize!` mutate in place. Put the tests in `test/test_irrep_termtable.jl`
next to the existing accumulation-route tests.

### 2b. W5 — abelian and non-abelian coupling have unrelated spellings

Adjacent in `research/progression.jl`, same physical shape:

```julia
couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0))   # non-abelian: must nest
couple(F.cd[1], F.c[2], F.cd[3], F.c[4])                              # abelian: folds variadically
```

This is the strongest candidate for change that the flow-reading exercise surfaced.
`research/interface-review.md` §7 (Option 4) sketches the fix: one variadic `couple` whose
`channels` are optional exactly when they are forced, plus a `couple_channels(ops...; to)` query
returning the legal tuples.

**Evidence it is a real gap:** `examples/multibody.jl` currently discovers the legal channels with a
`try`/`catch` loop over candidate `j` values. That loop only exists because there is no query.

**This one needs a decision before implementation.** The trade-off is on record in §7: a single
spelling costs you the self-documenting nested `to =`, and hides the caterpillar tree that the
present API deliberately exposes. Ask the author which they want; do not pick unilaterally. The
`couple_channels` query is a win either way and can land independently.

## 3. The example pages

`research/examples-progression.md` has the full tiered progression (§3), the suggested nine-page
layout (§4), the conventions and build commands (§5), and the W1–W9 watch-list (§6). Four questions
in its §7 need the author's input before the layout is final — most importantly whether measuring
operators (Tier 9: observables, correlators, charged operators) is in scope for OpSum's docs at all.

Findings already available from the progression run, which the pages should exploit:

* SU(2) Heisenberg is `D=3` where U(1) XXZ is `D=6` for the *same* operator — the symmetry payoff in
  one table.
* `truncrank(8)` → 8.1e-4 relative error on a power law, `truncrank(4)` → 0.77. The knob visibly
  bites; good material for the truncation page.
* `c†₂` compresses fine (`D=1`) but `jordan_mpo_tensors` **refuses** it. The Hamiltonian path and the
  measurement path genuinely diverge — the argument for Tier 9 being a page.
* `matrixunit` cannot reach the `U1 ⊠ SU2 ⊠ FermionParity` Hubbard site (the spin-½ sector has
  dimension 2), so `project` is the only route. Verify the error message is discoverable.
* W1 (repeating `lat`) turned out **milder** than predicted — usually one mention per block. Do not
  redesign for it.
* W3 (five routes to an on-site operator) resolves itself: the space dictates the choice. Reads as
  physics, not API sprawl.

## 4. Gotchas that will cost you a CI cycle

These all bit during PR #32. Each is cheap to avoid and expensive to rediscover.

* **Julia `min` is 1.10; local is 1.12.** `using OpSum: a_removed_name` is a hard `UndefVarError` on
  1.10 but only a **warning** on 1.12. So removing or renaming an export passes every local check
  and fails only the three CI `min` jobs. After touching `src/OpSum.jl`'s export list, grep the repo
  for the old name in `using OpSum:` lines, and run one test file under `julia +lts` (juliaup has
  1.10 as `lts` here).
* **`--project=test` does not work on 1.10.** `Project.toml` uses `[workspace]`, which is 1.11+. On
  1.10 go through `julia +lts --project -e 'using Pkg; Pkg.test()'`.
* **The `jld` daemon shares `Main` across requests.** A name you removed from the package can still
  resolve from a previous request's import, so a per-file `jld run` can pass on stale bindings. Use
  `jld restart`, or confirm with a fresh `Pkg.test()`, before believing a green per-file run.
* **`codecov/patch` will fail on this work.** It scores the whole diff, but coverage uploads only
  from the ubuntu + Julia 1 Tests job, and `examples/` and `benchmark/` are never executed by
  `Pkg.test()`. A PR that adds example pages therefore looks badly covered. Do not chase it: measure
  the real figure locally with `Pkg.test(coverage=true)`, then intersect `src/**/*.jl.*.cov`
  (`0` = uncovered, `-` = non-executable, merge all `<file>.*.cov` since there is one per worker)
  against `git diff -U0 <base> HEAD -- src/`. `codecov/project` and `ci-success` are the checks that
  matter; `codecov/patch` is not required.
* **Aqua's "Persistent tasks" check fails spuriously under `--code-coverage`**
  (`done.log was not created, but precompilation exited`). Not a regression.
* **`gh` quirks on this repo:** `gh pr edit` fails silently — get the PR body right on `create`.
  `--auto` does not gate on checks. The remote branch is auto-deleted on merge, so
  `git push origin --delete` afterwards errors harmlessly. Squash merge takes the *commit* message,
  so write a good one on a single commit.
* **Work in a worktree**, per the author's standing preference. The git stash stack is shared across
  worktrees — use a WIP commit rather than `git stash`.

## 5. Verification

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project research/progression.jl              # the reference, ~25 operators, prints a table
julia --project -e 'using Pkg; Pkg.test()'           # full suite
julia +lts --project -e 'using Pkg; Pkg.test()'      # the version CI catches import errors on
julia --project=docs docs/make.jl                    # runs Literate over every page AND every
                                                     # @example block in operators.md (~10 min)
```

Formatting is Runic, and it is a required check. It is not a project dependency:

```bash
julia --project=/tmp/runic -e 'using Pkg; Pkg.add("Runic"); using Runic; Runic.format_file("path.jl", "path.jl"; inplace=true)'
```

Acceptance for step 2: full suite green on **both** 1.12 and 1.10, Runic clean, `ci-success` green.
Acceptance for step 3: additionally `docs/make.jl` clean — no broken cross-references, no missing
docstrings, no failed `@example` blocks. Every new page must be added to the `EXAMPLES` vector in
`docs/make.jl`, in reading order (it sets the nav order).

## 6. Orchestration notes

The three steps have very different shapes; split them rather than running one long agent.

* **Step 2a** (`copy`/`zero`) is a contained ~20-line change plus tests. One worker.
* **Step 2b** (W5) is a **design decision first**. Do not delegate it to a worker — put the §7
  trade-off to the author, get an answer, and only then implement. The `couple_channels` query can
  be split off and done independently.
* **Step 3** parallelises by page, but the pages share `examples/common.jl` and the `docs/make.jl`
  nav list — so have workers write pages independently and integrate those two files centrally, or
  you will get conflicts on every page.
* Budget: each `docs/make.jl` run is ~10 min and each full suite ~7 min (1.10) to ~12 min (1.12).
  Verify in batches, not per page.
* The expensive thing in a page is the dense oracle: `instantiate` is exponential in `N`, and
  contracting an `N`-site MPO costs a fresh `ncon` specialisation per `(length, sectortype)`. The
  existing pages spend it only where a sign or a fusion coefficient could be wrong. Keep that
  discipline or page build time will balloon.

## 7. Open, not blocking

* Benchmark figures under `docs/src/assets/` were not regenerated after PR #32. The builders return
  `(h, lat)` now but compute the same operators, so the figures should be unaffected — unverified.
  `julia --project=benchmark benchmark/run.jl --sweep ci` if you want it confirmed.
* `benchmark/ShowcaseModels.jl` could grow to cover the new models (Kitaev, SU(2) Hubbard,
  non-uniform chains) so `test_showcase_models.jl` pins their bond dimensions. Author's call.
* `IrrepOperator` is still exported although the docs' first instruction is "never write one".
  Noted in `research/interface-review.md` §3; not addressed.
