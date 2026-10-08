# The high-level interface as it stands (meeting overview, 2026-10-08)

State of `main` at `5f246be`.
This is what a user writes today; [operator-bases.md](operator-bases.md) §3 describes the agreed change.

A model is written in four steps, and every model on the examples pages follows them:

```julia
V   = SU2Space(1 // 2 => 1)                          # 1. a physical space
S   = spin(V)                                        # 2. on-site operators
H   = opsum(FiniteChain(V, N),                       # 3. place, couple, bind to a lattice
            dot(S[i], S[i + 1]) for i in 1:(N - 1))
mpo = irrep_mpo(H)                                   # 4. compress (and emit tensors)
```

---

## 1. On-site operators

| call | returns | notes |
|---|---|---|
| `spin(V)` | `SiteOperator` | SU(2) vector operator on a single-irrep SU(2) space, normalised so that `dot(S[i], S[j])` is `S⃗ᵢ·S⃗ⱼ` |
| `spin_ops(V, up, dn)`, `spin_ops(V, sectors)` | `(; Sp, Sm, Sz)` | U(1)-graded spin; `sectors` in descending `m`, never inferred from the labels |
| `fermion_ops(V)`, `fermion_ops(V, vac, occ)` | `(; c, cd, n)` | spinless mode; the one-argument form only for `Vect[FermionNumber]` |
| `matrixunit(V, out, in)` | `SiteOperator` | `\|out⟩⟨in\|` between one-dimensional, multiplicity-one sectors |
| `scalarop(c, V)` | `SiteOperator` | `c · 𝟙` |
| `project(O, V)` | `SiteOperator` | any symmetric on-site `TensorMap`, expanded in the letters |
| `project(h)` | `LocalOperator` | a `K`-site symmetric `TensorMap` (optionally with a trailing charge leg), unplaced |

Operators combine with `+`, `-`, `*`, `/` by scalars.
There is no on-site product: build the product as a `TensorMap` and `project` it.
All builders are memoised, so calling them inside a term loop is cheap.

## 2. Placing and coupling

| call | meaning |
|---|---|
| `A[i]` | place a `SiteOperator` on site `i` → `Terms` |
| `B[i]`, `B[s₁, …, s_K]` | place a `LocalOperator` contiguously from `i`, or on explicit increasing sites → `Terms` |
| `couple(a, b; to = c)` | couple two operands to total charge `c` (default `unit(I)`); distributes over composite operands |
| `couple(a, b, c, …)` | variadic form, abelian sectors only (every intermediate charge is forced) |
| `couple_channels(a, …; to)` | list the intermediate-charge tuples a multi-operand coupling can take |
| `dot(a, b)` | the Cartesian scalar product `−√dim(c) · couple(a, b)`; either site order, any symmetry |
| unplaced `couple`/`dot` on `SiteOperator`/`LocalOperator` | build a `LocalOperator` on consecutive slots, to place later |

```julia
# abelian, any site order; couple supplies the fermionic sign
H = opsum(FiniteChain(V, N), -t * (couple(cd[i], c[i + 1]) + couple(cd[i + 1], c[i])) for i in 1:(N - 1))

# non-abelian, in site order, nested to name the channel
couple(couple(A[1], B[2]; to = c12), C[3])

# a projected bond, placed per bond
B = project(-t * (hop + hop'))
H = opsum(FiniteChain(Vh, N), (B[i] for i in 1:(N - 1)), (U * nupndn[i] for i in 1:N))
```

The rules a user has to know:

* Under an abelian symmetry operands may come in any order and `couple` inserts the braiding phase (the fermionic sign).
* Under a non-abelian symmetry `couple` must follow site order; reordering would need F-moves and throws.
* `couple` distributes over composite operands and drops pairs that cannot reach `to`; `dot` does not distribute.

## 3. The lattice-bound sum

| call | meaning |
|---|---|
| `FiniteChain(V, N)`, `FiniteChain(spaces)` | an open chain |
| `InfiniteChain(V)`, `InfiniteChain(spaces)` | a repeating unit cell; the terms are a generating set |
| `opsum(lat, terms...)` | a new `OperatorSum`, filled in one pass (iterables nest) |
| `opsum!(H, terms...)` | append in place; `H + x` copies |
| `expterm(t; decay, exitsite, string)` | an exponentially decaying channel (`ExpSum`), added with `opsum(lat, h, expterm(h; decay = λ))` |
| `H'` | hermitian conjugate (charge-neutral terms, no decaying channels) |
| `instantiate(H)` | dense `TensorMap` on a finite chain (the test oracle) |
| `chain_terms(H)` | the explicit term sum, decaying channels expanded on the finite chain |

```julia
forward = opsum(FiniteChain(V, N), -t * couple(cd[i], c[i + 1]) for i in 1:(N - 1))
H = forward + forward'

Hinf = opsum(InfiniteChain(V), dot(S[1], S[2]), expterm(dot(S[1], S[2]); decay = 0.4))
```

All four cells (finite / infinite × plain / decaying) share one signature: the lattice goes in `opsum`, and every consumer takes the `OperatorSum`.
The older `irrep_mpo(::Terms, lat)` forms throw a migration hint.

## 4. Compression and output

| call | returns | notes |
|---|---|---|
| `irrep_mpo(H[, alg])` | `FiniteMPO` / `InfiniteMPO` | destructures as `Ws, bondsectors` |
| `irrep_mpo_tensors(mpo, lat)` | `Vector{TensorMap}` | dense symmetric site tensors |
| `jordan_mpo_tensors(H[, alg])` | `Vector{SparseBlockTensorMap}` | Jordan form for MPSKit; finite chains without decaying channels only |
| `mpo_terms(mpo)` | `Terms` | reconstruct the operator from the reduced MPO |
| `islossless(H[, alg])` | `Bool` | `mpo_terms(irrep_mpo(H)) ≈ chain_terms(H)`; finite chains only |
| `mpo_tensormap(Ts)` | `TensorMap` | contract a chain of site tensors |

`alg` is `BipartiteAlgorithm()` (default, lossless, minimal for the sparsity pattern) or `SVDBondAlgorithm(trunc; sweep = IndependentSVD | SequentialSVD)`.

Handing over to MPSKit is a short adapter, since OpSum does not depend on MPSKit:

```julia
Ws = jordan_mpo_tensors(H)
to_jordan(W) = (O = MPSKit.jordanmpotensortype(spacetype(W), storagetype(W))(undef, space(W));
                foreach(((ix, v),) -> (O[ix] = v), nonzero_pairs(W)); O)
Hmpo = FiniteMPOHamiltonian(map(to_jordan, Ws))
```

---

## What works well

* **One container, one signature.** Lattice choice, decaying channels and finite/infinite never change how the user binds terms or calls the compressor.
* **Symmetry is the space, not a flag.** The same model code runs on `ℂ²`, U(1) or SU(2) by changing the space and the on-site operators.
* **Fermions need no Jordan–Wigner strings**, and abelian `couple` gets the sign right in either order.
* **Verification is built in**: `islossless`, `instantiate` and `mpo_terms` make every model checkable.

## Where it is thin or awkward

* **No named operator sets for common sites.**
  Hubbard, electron and spin-orbital sites are assembled from `matrixunit` sums or a projected `TensorMap`, and the SU(2) chemistry script needs its own Clebsch–Gordan code to get coefficients.
* **The basis is implicit.**
  How an operator is split into letters is not visible or controllable, and it decides the bond dimension (#38).
* **Non-abelian models must be written in site order**, with every intermediate channel named.
* **No bulk path for `N⁴` terms.**
  Each term is a `couple` call; the #38 script accumulates `(sites, letters, channels)` in a `Dict` to cope.
* **MPSKit hand-over is manual**: the `to_jordan` adapter, the `ComplexF64`-only output, and no Jordan form for infinite chains.
* **Exported but discouraged**: `IrrepOperator` is exported, though model code should never construct one.
