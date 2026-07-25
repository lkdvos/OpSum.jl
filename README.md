# OpSum.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://lkdvos.github.io/OpSum.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://lkdvos.github.io/OpSum.jl/dev/)
[![Build Status](https://github.com/lkdvos/OpSum.jl/actions/workflows/Tests.yml/badge.svg?branch=main)](https://github.com/lkdvos/OpSum.jl/actions/workflows/Tests.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/lkdvos/OpSum.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/lkdvos/OpSum.jl)
[![Code Style: Runic](https://img.shields.io/badge/code_style-%F0%9F%AA%A8_Runic-9558B2)](https://github.com/fredrikekre/Runic.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

OpSum.jl converts sums of symmetric quantum operators — Hamiltonians — into **exact**,
symmetry-reduced matrix product operators. The pipeline is: symbolic term algebra → flat term list
→ per-bond-sector compression → reduced MPO tensors.

Two things distinguish it. The compression is *lossless*: the resulting MPO reproduces every term
of the input exactly, rather than being an approximation with a truncation threshold. And the
symmetry is carried all the way through, so the bond indices are irreducible-representation
labels and the MPO tensors come out as symmetric `TensorMap`s.

## Installation

This package is not yet registered. It can be installed directly from GitHub:

```julia
julia> using Pkg: Pkg

julia> Pkg.add(url="https://github.com/lkdvos/OpSum.jl")
```

## Quick start

The SU(2) Heisenberg chain — nearest-neighbour spin-spin coupling on a spin-½ chain — end to end:

````julia
using OpSum
using OpSum: irrep_mpo, mpo_terms, spin
using TensorKit
using LinearAlgebra: dot

V = SU2Space(1 // 2 => 1)          # one spin-½ per site
N = 8
sites = fill(V, N)

S = spin(V)                        # the SU(2) rank-1 vector operator
H = sum([dot(S[i], S[i + 1]) for i in 1:(N - 1)])

Ws, sectors = irrep_mpo(H, sites)  # reduced bond matrices + per-bond charge sectors
````

The bond dimension: `sectors[b]` lists the irrep labels on the bond to the right of site `b`, so
its length is the number of symmetry-resolved indices, and the quantum-dimension-weighted sum is
what a symmetry-agnostic MPO would need.

````julia
bulk = 4
(reduced = length(sectors[bulk]), dense_equivalent = sum(dim(c) for c in sectors[bulk]))
````

Three multiplets — identity-in, identity-out, and one open spin-1 channel — where a dense MPO
needs five states.

The compression is exact, which `mpo_terms` verifies by reconstructing the original term sum:

````julia
back = mpo_terms(Ws, sectors)
Set(keys(back.terms)) == Set(keys(H.terms)) &&
    all(back.terms[k] ≈ H.terms[k] for k in keys(H.terms))
````

## Examples

The example pages build a range of models and measure how the bond dimension responds:

- [Shared utilities](https://lkdvos.github.io/OpSum.jl/dev/examples/common/) — naming alphabet
  letters, coupling composite operators, lattice geometry, and the verification helpers.
- [Spin chains](https://lkdvos.github.io/OpSum.jl/dev/examples/spin_chains/) — Heisenberg, XXZ and
  ``J_1``–``J_2``: how symmetry and interaction range set the bond dimension.
- [Long-range interactions](https://lkdvos.github.io/OpSum.jl/dev/examples/long_range/) —
  Haldane–Shastry and power laws, the one family whose bond dimension grows with ``N``.
- [Ladders and cylinders](https://lkdvos.github.io/OpSum.jl/dev/examples/ladders_and_cylinders/) —
  quasi-2D geometries, where the bond dimension tracks the circumference and not the length.
- [Multi-body interactions](https://lkdvos.github.io/OpSum.jl/dev/examples/multibody/) — three- and
  four-body couplings, and the fusion channels that label them.
- [Fermions](https://lkdvos.github.io/OpSum.jl/dev/examples/fermions/) — free chains and the
  Fermi–Hubbard model, with no Jordan–Wigner strings anywhere.

Construction time and bond dimension across system size are measured by the benchmark harness in
`benchmark/` and plotted with `scripts/plot_benchmarks.jl`.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

