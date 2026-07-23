# Migration: retire `GlobalOp`, port the dense track to the term representation

Companion / follow-up to [`term-algebra-redesign.md`](./term-algebra-redesign.md). That doc
migrated the **ITO track** from an expression tree to a flat term-sum and explicitly deferred the
dense track ("`GlobalOp`/`LocalOp` → … left untouched; dense migrates in a later follow-up"). This
is that follow-up: the scoping to retire `src/operators/globalalgebra.jl`.

## Goal / non-goals

- **Goal:** delete `globalalgebra.jl` (`GlobalOp`, `SiteOp`) by giving the dense/Pauli pipeline a
  term-based front end, converging its representation with the ITO track's `TermKey`/`TermSum`.
- **Survives:** `operatoralgebra.jl` (`LocalOp`) stays — it is the on-site letter/alphabet wrapper
  both tracks depend on, and `operatorstrings` (its Sum/Prod→normal-form expander) is reused.
- **Non-goal (unchanged):** unifying the two *bond-optimization sweeps*
  (`graphbuilding.jl`'s `TermTable` sweep vs `irreptermtable.jl`'s per-sector `_irrep_bipartite`).
  They can converge later; this migration only touches the front end and the table *builder*.
- **Non-goal:** operator multiplication under symmetry (already deferred on the ITO track). Dense
  on-site `*` is *not* deferred — it is ordinary `LocalOp` multiplication (trivial sector).

## Current reality (what actually depends on `GlobalOp`)

The dense pipeline is `GlobalOp → TermTable → bipartite/SVD → MPO`. Crucially, **the bond-opt core
is already `GlobalOp`-free**: `mpo_bond_optimizations(vertices, tt::TermTable, alg)`
(`graphbuilding.jl:47`, `:303`) reads only `_op_at(tt,…)`, `tt.coeffs`, `nterms`, `nvertices`.
`GlobalOp` survives in exactly three places:

1. **Front-end arithmetic / construction** — `globalalgebra.jl`: `op[sites…]` (`getindex`→`SiteOp`),
   `+`, `*`, `scale`, `opsum`, `simplify`, `show`. This is how a user writes `2.0*X[1]*Z[2] + Y[3]`.
2. **Table builder** — `TermTable(vertices, ex::GlobalOp)` (`termtable.jl:58`) via `_collect_terms!`
   + `operatorstrings`. Note `_collect_terms!` only accepts `Sum`/`SiteOp` variants and `error`s on
   `Prod`/`Pow`/`Fun` — i.e. the MPO path already requires a *Sum-of-SiteOps normal form*.
3. **Dense oracle** — `instantiate(ex::GlobalOp, sites)` (`globalalgebra.jl:40`), the correctness
   reference in `test/instantiate.jl` and the MPO round-trip tests.

`Prod`/`Pow`/`Fun` in the `@sumtype` are transient scaffolding: they exist only so `instantiate`
can densify lazy expressions; `*` already normalizes to `SiteOp`s for the MPO path.

Consumers to migrate: `test_mpo_bipartite.jl`, `test_mpo_svd.jl`, `test_mpo_termtable.jl`,
`test_termtable.jl`, and the `GlobalOp` testset in `test/instantiate.jl`.

## Target representation

Reuse the ITO term machinery, generalized over the **letter type** `L` (today it is hardwired to
`IrrepOperator{I}`). The dense case is the **trivial-sector** instance:

```julia
struct Term{L, I<:Sector, S}          # today: TermKey{I,S} with ops::Vector{IrrepOperator{I}}
    sites::Vector{S}                  # active sites, sorted & unique
    ops::Vector{L}                    # one alphabet letter per active site
    tree::FusionTree{I,K}             # caterpillar over the K charges → total
end
```

Why this works for dense: after normalization each active site carries a **single alphabet letter**
(`operatorstrings`/`_local_terms` already expand `Sum`/`Prod` into `(coeff, [letter…])` strings), so
dense terms have the same `one-letter-per-site` shape as ITO terms. With `I = Trivial`, every charge
is `Trivial()` and the caterpillar `tree` is unique/trivial — the `FusionTree` field is a harmless
no-op, and `Term` degenerates to ITensor-`OpSum`-style `(coeff, [(site, letter)…])`.

Letter interface (the generic hook; extend `OperatorBasis` or a small trait):
- `charge(op::L)::I` — `op.c` for ITO, `Trivial()` for dense.
- coupling: `_leaftree`/caterpillar builders keyed on `charge` (already exist for ITO; dense gets the
  trivial tree).
- on-site combine for `*`: `LocalOp` multiplication for dense (available); deferred/error for ITO
  (already the case).

Container: reuse `TermSum` (a `Dictionary{Term, T}`), generalized to `TermSum{L,I,S,T}`. `+`/`scale`
are already dictionary merge/accumulate (`irrepalgebra.jl:193,206`).

## Design decisions to settle (recommendations)

1. **Generalize the existing `TermKey`/`TermSum` vs. add a parallel dense `Term`.**
   *Recommend generalizing* `TermKey{I,S}`→`TermKey{L,I,S}` (ops `Vector{L}`) and
   `TermSum{I,S,T}`→`TermSum{L,I,S,T}`. One representation, matches the redesign's north star, and
   dense/ITO diverge only in the letter type. Risk: touches green ITO code — mitigated by the ITO
   test net (step 5). If that risk feels too high, fall back to a parallel `DenseTerm` sharing shape
   but not code (achieves the goal, leaves unification for later).
2. **Where the letter interface lives.** Add `charge`, `combine`, and a coupling hook to the
   `OperatorBasis` interface (`operatorbasis.jl`) so both `IrrepOperator` and `PauliOperator` slot
   in — no second rewrite when dense arrives.
3. **`Prod`/`Pow`/`Fun` fate.** The MPO path never needs them. Keep them alive *only* if the
   `instantiate` oracle tests exercise lazy products/functions that aren't easily expressed as a
   term-sum. Audit `test/instantiate.jl`; likely they can be dropped or moved to a small standalone
   `LazyOp` helper decoupled from the MPO front end.
4. **Preserve the dense normal-form guarantee.** `_collect_terms!` asserts sorted/unique sites and
   drops identities; the `Term` builder must keep those invariants (they are load-bearing for
   `_op_at` in the sweep, which assumes ascending zero-padded sites).

## Migration steps (each ends green + Runic)

1. **Letter interface.** Add `charge(op)`, on-site `combine(a,b)`, and coupling hooks to
   `OperatorBasis`; implement for `IrrepOperator` (trivial: forward to `.c`) and `PauliOperator`
   (`Trivial()` + `LocalOp` `*`). Pure addition — no behavior change, full suite stays green.
2. **Generalize the term types.** `TermKey{I,S}`→`TermKey{L,I,S}`, `TermSum{I,S,T}`→
   `TermSum{L,I,S,T}`, and `ITOTermTable` construction, keyed on `charge(op)` instead of `op.c`.
   The ITO track re-specializes to `L = IrrepOperator{I}`; **its tests are the safety net** and must
   stay green with no assertion changes.
3. **Dense front end.** Provide dense `op[sites…]`, `+`, `*` (disjoint = concat; on-site = `combine`),
   `opsum`, `simplify` producing `TermSum{PauliOperator,Trivial,S,T}`. Lift `operatorstrings`'
   Sum/Prod expansion into term construction (mirror `_local_terms`/`getindex` from
   `irrepalgebra.jl:234`).
4. **Table builder + oracle.** Add `TermTable(vertices, ts::TermSum)` (replacing the `GlobalOp`
   overload) and `instantiate(ts::TermSum, sites)` for the dense oracle (mirror
   `irrepalgebra.jl:347`). The bond-opt sweeps are untouched — they already consume `TermTable`.
5. **Port dense tests.** Rewrite `test_termtable`, `test_mpo_bipartite`, `test_mpo_svd`,
   `test_mpo_termtable`, and the `instantiate` `GlobalOp` testset onto the new front end, keeping
   every assertion (same physics, new surface).
6. **Delete `globalalgebra.jl`.** Remove the `include`, `GlobalOp`/`SiteOp`, and the now-dead
   arithmetic. Drop `Prod`/`Pow`/`Fun` per decision (3). Re-export `opsum`/`simplify` from their new
   home.

## Risks

- **Destabilizing the green ITO track** (step 2 edits shared types). Mitigation: the ITO tests are
  the net; make step 2 a pure generalization (`L` defaults recover today's types).
- **Dense on-site `*` semantics.** `X[1]*Z[1]` must route to `LocalOp` `*` (→ `XZ`), `X[1]*Z[2]` to a
  two-site term. Cover both in tests before deleting `GlobalOp`'s `*`.
- **`instantiate` oracle coverage.** If any test relies on lazy `Prod`/`Pow`/`Fun` densification,
  either express it as a term-sum or retain a minimal `LazyOp`. Audit first (decision 3).
- **`simplify` semantics.** In the term-sum world `simplify` is largely free (dictionary keys are
  canonical); confirm no test depends on `GlobalOp`-specific simplification behavior.

## What dies / what survives

- **Dies:** `src/operators/globalalgebra.jl` in full (`GlobalOp`, `SiteOp`, its arithmetic,
  `instantiate(GlobalOp)`, `simplify(GlobalOp)`); the `TermTable(…, ::GlobalOp)` overload; likely
  `Prod`/`Pow`/`Fun`.
- **Survives:** `operatoralgebra.jl`/`LocalOp` (letter wrapper, both tracks), `operatorstrings`,
  `TermTable` + both bond-opt sweeps, `bipartite.jl`, `state_machines.jl`, `paulioperators.jl`.
- **Generalized (not deleted):** `TermKey`/`TermSum`/`ITOTermTable` gain a letter type parameter.

## Validation

- ITO suite green throughout (the refactor's correctness net for shared types).
- Dense suite ported and green (same assertions).
- Full suite + Runic at each step; `globalalgebra.jl` gone only after all dense tests pass on the new
  front end.
