# Add exponentially decaying interactions to the infinite MPO construction

*Handoff prompt for a fresh Claude Code session started in this repo. The design it implements is
`research/infinite-mpo.md` §7; the machinery it builds on is `research/persistent-graph-mpo.md` §2.*

## Objective

Support terms of the form

```math
\sum_{i<j} \lambda^{\,j-i-1}\; A_i \, S_{i+1} \cdots S_{j-1} \, B_j ,\qquad |\lambda| < 1
```

on an **infinite chain with a repeating unit cell** (`irrep_mpo(H, ::InfiniteChain)`, already built).
These are the interactions whose Jordan-MPO signature is a scalar `λ` on the **diagonal** of a bond
channel. They cannot be enumerated as a flat term list — the sum is infinite — so they need a new
primitive and one genuine change to how suffix classes are named.

Finite-range infinite MPOs already work and are fully tested; **do not regress them**. The finite
(open-boundary) pipeline must be untouched behaviourally.

## Required reading (in order)

1. `research/infinite-mpo.md` — **your primary spec.** §7 is the design for this task. §1 is the
   survival table, §3 explains the canonicalisation and *why* it is load-bearing (you will be extending
   it), §4 the pending↔started collision and the trailing-coefficient consequence, §5 the window
   construction and identity-channel detection, §6 what the tests actually pin.
2. `research/persistent-graph-mpo.md` §2 — the sweep you are extending. §2.1 (interned suffix classes)
   and §2.2 (lazy insertion + the collision) are the parts this task changes.
3. `src/operators/irrepgraph.jl` — `ITOGraph`, `_suffix_ids`, `_rel_suffix_ids`, `_rdesc`,
   `_signature!`, `_suffix_merge!`, `_promote_pending!`, `_canonicalise_rights!`,
   `_canonicalise_bond!`, `_vc_component`, `_at_site!`.
4. `src/operators/infinitechain.jl` — `InfiniteChain`, `translate`, `unitcell_terms`, `window_terms`,
   `maxspan`.
5. `src/operators/infinitegraph.jl` — `InfiniteMPO`, `_infinite_window`, `_fixedpoint_cell`,
   `_identity_channels`, `mpo_terms_window`, `contract_open`, `_canonform`.
6. `src/operators/irrepmpo.jl` — the output contract, `mpo_terms` (with its `leftidx`/`rightidx`
   keywords), `irrep_mpo_tensors` / `_mpo_tensors`.
7. `CLAUDE.md`, `test/test_infinite_mpo.jl`, `test/test_infinite_graph.jl`.

## The key idea: a geometric channel is a suffix class with a self-loop

Do **not** build a separate, bolted-on block of "exponential channels". Make a geometric channel an
*ordinary right vertex* whose interned name is self-referential:

```
self edge:  λ · passthrough  →  itself
exit edge:  B                →  the exhausted class
```

Then the existing machinery does the compression, and these all fall out rather than being special-cased:

| case | outcome | mechanism |
|---|---|---|
| same `λ`, same exit, different entry | merge | equal suffix class (`_suffix_merge!`) |
| same `λ`, same entry, different exit | merge | shared left vertex, covered by the min vertex cover |
| different `λ` | stay distinct | correct — the channels are linearly independent |

It also needs **no forcing into the bond basis**: the channel regenerates itself as the covered-left
vertex `(link = g, key = λ·passthrough)`, which *is* the fixed point — the same structure that already
keeps the start and done channels alive at every bond (`infinite-mpo.md` §5).

## The one change of kind

Bottom-up hash-consing cannot name a cyclic tail. Both `_suffix_ids` (position-keyed) and
`_rel_suffix_ids` (shape-keyed) terminate *because* they cons onto an already-interned tail, and a
self-reference has none.

The replacement is **partition refinement** (Hopcroft–Moore) — the coinductive dual: start with all
classes in one block, split by `(next-site key, successor class)` until stable, identifying classes up
to bisimulation. Note that `_suffix_merge!` is already one refinement step per bond; it just refines
*by a precomputed name* rather than *toward a fixed point*. Getting this right is the bulk of the task.

Both existing names have consumers that constrain the design — read them before replacing anything:
`_signature!` (per-bond merge, and `pendbysig` keys on the *absolute* form), and `_rdesc` (the
translation-invariant canonical order from §3, which the unit cell's closure depends on). A geometric
class needs a canonical name in the `_rdesc` sense too, or the cell will stop closing.

## What to build

1. **The primitive.** Something like `expterm(A, B; decay = λ, string = 𝟙, to = unit(I))`, held
   *alongside* the generating `TermSum` — it cannot live in `ITOTermTable`'s `K×M` matrices. That means
   a container (extend `InfiniteChain`'s input, or a small wrapper type) plus a decision about how
   `irrep_mpo` receives both. Validate `|λ| < 1` and reject `λ = 1` with a message saying it is not
   summable.
2. **Cyclic class naming** — the partition-refinement replacement above, with geometric classes
   participating in `_suffix_merge!` and in the canonical order.
3. **Seeding and the sweep.** A geometric channel is live at *every* bond (it never starts and never
   finishes, like the identity channels), so it does not go through `pend_at` / lazy insertion the way a
   finite term does. Work out where it enters and make sure `_promote_pending!` still fires correctly
   for finite terms whose tails now coexist with geometric ones.
4. **Fixed-point detection.** `_fixedpoint_cell` compares cells entry for entry; check it still
   converges with a geometric channel present (the diagonal entry is `λ·passthrough` at every bond, so
   it should, but the residual-coefficient settling argument in §5 needs re-checking).

## Watch-outs

- **The diagonal must carry the pass-through letter weighted by λ**, never be an *empty*
  `SiteOperator`. Tensor assembly iterates `pairs(localop)`, so a letter-less entry is silently dropped
  and you would get a correct-looking reduced MPO and a wrong tensor. `SiteOperator` makes this hard to
  trip over — the bare identity is the `passthrough` sentinel letter, so `scalarop(λ, I)` already is
  `λ · passthrough` — but λ still has to ride as an *edge weight* rather than be folded away.
- **`SiteOperator`'s `==` is order-sensitive.** It compares the `letters`/`coeffs` vectors in order and
  `+` accumulates in insertion order, so two separately built copies of the same entry can disagree.
  Compare through `_canonform` (`infinitegraph.jl`), which sorts by letter. Making `SiteOperator`
  canonically ordered would let `_canonform` go away and is a listed follow-up.
- **Charge neutrality is assumed** throughout the infinite path (`unitcell_terms` enforces
  `total == unit(I)`), because the running bond charge is referenced to the left vacuum. A geometric
  channel must respect the same invariant: `A ⊗ (string…) ⊗ B` fusing to `unit(I)`.
- **`mpo_terms_window` is a sandwich, not an equality** (§4/§6) — a term's coefficient can sit up to `R`
  sites past its own support. With a geometric channel there is no finite `R`, so the completeness half
  of the existing `faithful` helper in `test/test_infinite_mpo.jl` needs rethinking, not just
  re-parameterising.
- Do not touch the **running-bond-first** fusion coupler ordering in `irrep_mpo_tensors`; charge-first
  flips signs at antisymmetric vertices for `K ≥ 3`.

## Verification

The honest oracle is an **explicit truncation**: replace the geometric channel by the finite term list
`Σ_{r=1}^{R_max} λ^{r-1} couple(A_1, B_{1+r})`, build *that* as an ordinary finite-range infinite MPO,
and compare on a window. Every term within the window must match coefficient for coefficient, and the
bond dimension of the geometric version must be `R_max`-independent while the truncated one grows.

Also:
- `contract_open` (`infinitegraph.jl`) for a dense check on a small cell against `instantiate`.
- Merge behaviour from the table above, as direct bond-dimension assertions: two channels sharing
  `(λ, string, exit)` must cost one channel, differing `λ` must cost two.
- `julia --project -e 'using Pkg; Pkg.test()'` — 3478 tests currently pass; none may break.
- Targeted: `julia --project=test test/test_infinite_mpo.jl`, `test/test_infinite_graph.jl`,
  `test/test_irrep_graph.jl` (the last one guards the shared sweep).

## Out of scope

Jordan blocks (polynomial × exponential decay); sum-of-exponentials *fitting* for power laws — worth
designing the primitive so a fit can be layered on top, but do not implement the fit;
`SVDBondAlgorithm`, which does not extend to infinite chains at all.

## Deliverable

Working primitive + sweep support, tests in the existing standalone / `ParallelTestRunner` style, every
touched file formatted with Runic (`CLAUDE.md` has the incantation), and `research/infinite-mpo.md` §7
rewritten from a design sketch into a description of what was actually built — including whatever the
design got wrong.
