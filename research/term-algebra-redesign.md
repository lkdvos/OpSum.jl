# Redesign: term-list global algebra

Companion to [`symmetric-operators-ito-design.md`](./symmetric-operators-ito-design.md) and
[`symmetric-operators-implementation-plan.md`](./symmetric-operators-implementation-plan.md).

## Context

Phases 1–3 (ITO alphabet → symbolic algebra → charge-augmented trie) are committed and green
(794 tests). Two warts emerged: coupling (`CoupledOp`) lives *outside* the `GlobalOp` sum-type,
and the `GlobalOp` `Sum/Prod/Pow/Fun` expression tree is transient scaffolding that the pipeline
(`operatorstrings`/`build_trie!`/`fusion_resolve`) immediately flattens into a normal form
(`FusedTerm`). The realization: **the normal form should be the representation.** A Hamiltonian is
a sum of terms; each term is a (sparse, sited) fusion-resolved `FusedTerm`.

Decisions (agreed):
- **Generic over the letter type, ITO-first.** Build a `Term` representation generic enough that
  the dense/Pauli case is the trivial-sector special case, but prove it on the ITO/symmetric track
  first. The existing tested dense pipeline (`GlobalOp`/`LocalOp` → trie → bipartite/SVD) is left
  **untouched**; dense migrates in a later follow-up.
- **Now, before Phase 4/5**, so per-sector bond optimization and MPO assembly build on the clean
  representation and `CoupledOp` never propagates.

Non-goal (unchanged): operator *multiplication under symmetry* (on-site fusion / F-moves) is
inherent complexity that lives in `*` regardless of representation — still deferred.

## Target representation

A **sparse, sited** term (elevates Phase-3 `FusedTerm`, adds explicit sites, drops full-lattice
padding):

```julia
struct Term{L, I<:Sector, S, T}
    sites::Vector{S}          # active sites, sorted & unique
    ops::Vector{L}            # letter on each active site (L = IrrepOperator{I} for now)
    tree::FusionTree{I,K}     # caterpillar over the K active charges → total charge
    coeff::T                  # reduced coefficient
end
```

`Term` is a **hashable, orderable key** (`==`/`isless`/`hash` over `(sites, ops, tree)`), exactly
like `ITOKey`/`IrrepOperator` already are. Genericity over `L`: `charge(op::L)` and the coupling
representation are supplied by the letter type — for `IrrepOperator` the charge is `op.c` and
coupling is the `FusionTree`; a future dense letter has trivial charge and a trivial tree, so
`Term` degenerates to `(coeff, [(site, op)…])` (ITensor-`OpSum`-style).

## Global container = reuse `Sum`

The global algebra is a **sum over `Term` keys**: a `Dictionary{Term, T}` accumulating
coefficients — structurally identical to the existing `Sum{T,O}` (`abstractoperators.jl`). Two
options:
1. Make `Term <: SymbolicAlgebra{T}` and reuse `Sum{T, Term}` verbatim (gets `+`/`scale`/`inner`
   via `VectorInterface` for free).
2. A lightweight parallel `Dictionary{Term,T}` container if we don't want to overload
   `SymbolicAlgebra` semantics onto a term-key.

**Recommend (1)** if `Sum`'s `O <: SymbolicAlgebra{T}` bound is comfortable to satisfy; else (2).
Either way `+`/`scale` reduce to dictionary merge/accumulate — no new arithmetic engine.

## Operations

- `op[sites...]` → builds `Term`(s); a `LocalOp` `Sum` on a site **distributes** into several terms
  (already implemented as `_local_terms` in `irreptrie.jl` — lift it here).
- `couple(a, b; to)` / `a · b` → a single `Term` with the fused `tree` and reduced `coeff`
  (`-√dim(c)` singlet convention). **Retires `CoupledOp`.**
- `+`, `scale` → `Sum{T,Term}` merge/accumulate.
- `*` → distribute (cartesian product of terms); on-site overlap needs fusion recoupling →
  **deferred** (abelian / disjoint-site only first), same as today.
- The trie is a **derived index**: iterate `pairs(terms)`, spell the `ITOKey` path per term, insert.
  `irrep_trie` stays but consumes the term container directly; `fusion_resolve` is subsumed into
  `Term` construction (coupling is resolved when the term is built, not in a later pass).

## Migration steps (ITO track only)

1. Add `Term` + the letter-algebra hooks (`charge`, coupling accessors); make it a hashable key.
2. Choose the container (reuse `Sum` vs parallel dict); wire `+`/`scale`.
3. Rework Phase-2 `couple`/`·` (`irrepalgebra.jl`) to return `Term`/`Sum{T,Term}`; delete
   `CoupledOp`. Keep `spin`/`scalarop` and `LocalOp` instantiation.
4. Rework Phase-3 (`irreptrie.jl`): `Term` carries the caterpillar tree at construction;
   `irrep_trie(terms, sites)` builds from the container; keep `ITOKey`, `bondcharges`,
   `trie_terms`, pass-through identity. Fold `fusion_resolve` into `Term` construction.
5. Port `test_irrep_algebra.jl` + `test_irrep_trie.jl` to the new API; keep every assertion
   (dense-oracle couplings, block-diagonality, round-trip). **These tests are the safety net.**

**Untouched:** `GlobalOp`/`LocalOp`-as-tree for the dense path, `operatorstrings`, `build_trie!`,
`Trie`, `bipartite.jl`, `graphbuilding.jl`, `state_machines.jl`, `paulioperators.jl`, and all
existing dense tests (315 baseline). `LocalOp` remains the on-site builder for both tracks.

## Impact on just-committed Phase 2/3 code

This *reshapes* Phase 2/3 representation while preserving their functionality:
- `CoupledOp` removed; `couple` semantics preserved on `Term`.
- `FusedTerm` → `Term` (sparse + sited); `bondcharges`/`vertexlabels`/`trie_key`/`ITOKey`/
  pass-through carry over largely intact.
- All Phase 2/3 tests must stay green after porting (same physics, new surface).

## Risks / decisions to settle during implementation

1. `Sum{T,Term}` reuse vs parallel container (the `SymbolicAlgebra{T}` bound question).
2. Where the generic letter-algebra interface lives (extend `OperatorBasis`, or a new trait) so
   dense can slot in later without a second rewrite.
3. Sparse-term site bookkeeping: sorting, disjoint-site `couple` across arbitrary positions,
   idle-site pass-through only materialized at trie/instantiation time.
4. Keep the `couple` reduced-coefficient convention identical (still unverified end-to-end until
   Phase 5's dense oracle — do not silently change it during the refactor).

## Validation

- **Every Phase 2/3 test ported and green** (they encode the dense-oracle couplings, block-
  diagonality, and round-trip — the refactor's correctness net).
- **Dense baseline untouched and green** (315 tests) — proves the refactor is isolated to the ITO
  track.
- Full suite green end-to-end; format with Runic.
- Only after this: proceed to Phase 4 (per-sector bond optimization) on the term/trie foundation.
