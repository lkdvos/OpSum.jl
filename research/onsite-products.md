# On-site products and powers of ITOs

*Why the symbolic `Prod`/`Pow` scaffolding was removed, and what it would take to add it back
deliberately.*

## What was there

`LocalOp{T,A}` was a `LightSumTypes.@sumtype` over seven variants: `T` (scalar), `A` (alphabet
letter), `Sum`, `Prod`, `Pow`, `Kron` and `Fun`. Of those, the ITO pipeline only ever constructed
`T`, `A` and `Sum`:

- `Prod` / `Pow` were reachable but threw — `instantiate` raised
  `"on-site products/powers of ITOs (fusion recoupling) are deferred"`, and `_local_terms` raised
  `"cannot resolve on-site variant … (products/powers deferred)"`.
- `Kron` / `Fun` were never constructed anywhere outside the sum-type machinery itself.
- `simplify` handled all seven and had no callers and no tests.

The scaffolding cost a dependency (LightSumTypes), ~530 lines across `abstractoperators.jl` and
`operatoralgebra.jl`, an Aqua unbound-type-parameter failure from the generated constructor, and a
60-line `inner` with `error("TBA")` fallthroughs. It has been replaced by a concrete on-site type.

## What still works today

**Products of on-site operators are fully supported — just not symbolically.** Build the product as
a `TensorMap` and project it:

```julia
V  = SU2Space(1//2 => 1)
Sz = ...                       # any TensorMap  V ← V
project(Sz * Sz, V)            # → an on-site operator in the ITO alphabet
```

`project` is exact and complete (see `irrepprojection.jl`: the candidate basis is orthogonal with
the closed-form diagonal `inner(E,E) = dim(tot)/Π dim(c_k)`, and the candidate count equals the
dimension of the target homspace), so nothing is lost by going through the dense representation. The
cost is `O(d^4)`-ish for a single site, paid **once** — the intended usage is to build on-site
operators outside any loop and then place them with `[i]` / `couple`.

So the gap is one of *ergonomics and cost*, not of capability: you cannot write `Sz^2` and have it
stay symbolic, and if you needed the product of two operators that are themselves large symbolic
sums you would pay a dense round-trip.

## What adding it back would require

The obstruction is genuine, not incidental. An ITO letter `O_{c,n}` materialises as

    O_{c,n} :  V  ←  V ⊗ Vect[I](c => 1)

i.e. an operator carrying a dangling charge-`c` leg. The composition of two such operators on the
same site carries **two** charge legs, `c₁` and `c₂`, which have to be fused to a definite total `c`
before the result is again a single ITO. Expressing the result back in the alphabet of charge `c` is
exactly the recoupling problem: the reduced matrix elements of a product of two irreducible tensor
operators acting on the same space are related to those of the factors by Racah (6j) recoupling
coefficients, not by a scalar.

Concretely, an implementation would need:

1. **A recoupling step in the on-site algebra.** For `UniqueFusion` sectors (`Trivial`, `U₁`, `ℤₙ`,
   `FermionNumber`) this degenerates — charges add, there is one channel, and the product of two
   letters is a single letter times a scalar. That case is cheap and could be done first.
2. **6j / F-symbol machinery for the non-abelian case.** TensorKit exposes the F-symbols, and
   `WignerSymbols` is already a transitive dependency via TensorKit, so the data is reachable. The
   work is in getting the conventions right and testing them, not in obtaining the symbols.
3. **A normalisation audit.** The alphabet is normalised so that `inner(O,O) = 1` under TensorKit's
   qdim-weighted inner product (`instantiate` divides by `sqrt(dim(s))`). Any product rule has to be
   stated in that normalisation, and the natural recoupling formulae are usually quoted in a
   different one. This is where a naive port would silently pick up a `sqrt(dim(c))` per vertex.
4. **A decision about `Pow`.** Once `Prod` is right, `Pow` is `power_by_squaring`, but for a
   *finite-dimensional* site the powers are not independent (`Sz^(2s+1)` is a combination of lower
   powers), so a symbolic `Pow` that never simplifies would grow without bound. Either simplify
   against the site dimension or keep `Pow` lazy and resolve it at `instantiate` time.

## Recommendation

Add it only when there is a model that actually needs it, and add it as a *concrete operation on the
concrete on-site type* (`OnsiteOp`), not as a lazy expression-tree variant. The lazy variant is what
made the old design costly: it forced every consumer (`_local_terms`, `instantiate`, `inner`,
`simplify`) to branch over cases that could not occur, and pushed the "not implemented" error to the
far end of the pipeline instead of the call site.

The abelian case (item 1) is a self-contained, testable increment and is the natural starting point.
