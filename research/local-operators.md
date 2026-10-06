# Unplaced multi-site operators and a lattice-carrying `OperatorSum` — design note

Decided and implemented 2026-10-01 on the `local-operators` branch (PR #37). Builds on
`interface-review.md` (§1, §10–12; this note reverses part of §12, deliberately).

## 1. Why

`Term` is `(sites, keys, coeff)`, so a K-site operator could only exist *placed*: `project(h, sites)`
took the sites, `couple`/`dot` only accepted placed operands, and every model re-projected the same bond
block once per bond. `TermSum` (§1 of the review) failed because it did not cover infinite chains or
`expterm`; the replacement container has to.

## 2. Decisions

| question | decision |
|---|---|
| unplaced K-site operator | `LocalOperator{I}`: a `Terms` bag on slots `1:K` plus `K`; `SiteOperator` stays separate (it is the sweep's bond-matrix entry) |
| terms on a subset of `1:K` | no; every term has exactly `K` keys (invariant checked in the constructor) |
| scalars / identity factors | the `passthrough` letter holds the slot; placement drops it; an all-pass-through term places as the `K = 0` scalar |
| `couple`/`dot`/`couple_channels` on unplaced operands | yes, by lowering onto consecutive slots and running the placed code (no reordering, no R-symbol) |
| placement | `B[i]` contiguous, `B[s₁, …, s_K]` strictly increasing; a monotone relabelling, exact for any symmetry (a gap site is a pass-through; fermionic strings come from the bond charge) |
| where spaces enter | `OperatorSum(lat)`: lattice + `Terms` + `ExpSum` (absorbs `MixedSum`); checked when terms *enter* |
| `H += x` | allowed (copies); `opsum!`/`opsum(lat, …)` are the linear routes |
| `(h, lat)` forms | removed; one migration hint, `irrep_mpo(::Terms/Term/ExpSum, …)` |
| `H'` | on the container only (needs the spaces), via the per-shape memoised `_adjoint_terms` |

## 3. What landed

* `src/operators/algebra/localoperator.jl`: `LocalOperator`, `nsites`, placement, `+ - * /`, `B ± α`,
  `≈`, unplaced `couple`/`couple_channels`/`dot`. `project(h)` returns a `LocalOperator`;
  `project(h, sites)` is `project(h)[sites...]`. Every operand after the first in an unplaced `couple`
  must be single-slot.
* `src/operators/operatorsum.jl`: `OperatorSum`, `opsum!`, `opsum(lat, …)`, insertion validation
  (sector, letters, site range; on an `InfiniteChain` neutrality and no `K = 0`), `+`, scalar `*`/`/`,
  `H'`, `instantiate(H)`, `chain_terms(H)`. `irrep_mpo`, `islossless` and `jordan_mpo_tensors` take an
  `OperatorSum`; the four finite/infinite × plain/decaying cells share one signature. "One
  representative per translation class" is checked in `unitcell_terms` when the MPO is formed.
* Fixes needed on the way: `project` verifies via `_instantiate_terms` (no public `instantiate(::Terms, …)`);
  `unitcell_terms` canonicalises first, so a term written twice is one term (needed for `H + H'` on an
  infinite chain).
* Examples, benchmarks and `research/progression.jl` are migrated; per-bond `project` loops became
  `b[i]`.

## 4. Deferred

* Lazy block-level adjoint `B'` (a pending second bag; `couple`/`dot`/`expterm` must refuse it).
* Spaces on `LocalOperator` (eager `B'`, `instantiate(B)`, earlier validation).
* Terms on a subset of `1:K`.
* Non-monotone placement `B[j, i]` (needs R-/F-symbols).
* Multi-slot later operands in unplaced `couple` (tree-structured, `via`).
* `irrep_mpo_tensors(mpo, lat)` reading the lattice off the MPO.
* `H'` on a container holding channels (throws).
* `islossless` on infinite chains (throws; no exact equality to test, a window sandwich was dropped to
  keep the PR small).
