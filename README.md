# OpSum.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://lkdvos.github.io/OpSum.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://lkdvos.github.io/OpSum.jl/dev/)
[![Build Status](https://github.com/lkdvos/OpSum.jl/actions/workflows/Tests.yml/badge.svg?branch=main)](https://github.com/lkdvos/OpSum.jl/actions/workflows/Tests.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/lkdvos/OpSum.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/lkdvos/OpSum.jl)
[![Code Style: Runic](https://img.shields.io/badge/code_style-%F0%9F%AA%A8_Runic-9558B2)](https://github.com/fredrikekre/Runic.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

OpSum.jl converts sums of symmetric quantum operators — Hamiltonians — into **exact**, symmetry-reduced matrix product operators.
The pipeline is: symbolic term algebra → flat term list → per-bond-sector compression → reduced MPO tensors.

Two things distinguish it.
The compression is *lossless*: the resulting MPO reproduces every term of the input exactly, rather than being an approximation with a truncation threshold.
And the symmetry is carried all the way through, so the bond indices are irreducible-representation labels and the MPO tensors come out as symmetric `TensorMap`s.

```julia
using OpSum, TensorKit

B = spin_basis(SU2Irrep)
S = spin_ops(SU2Irrep)
H = opsum(FiniteChain(B, 32), S.S_exchange[i, i + 1] for i in 1:31)
W = mpo(H)                       # exact, symmetry-reduced MPO
map(length, W.bondsectors)       # reduced bond dimensions
```

## Installation

Not yet registered:

```julia
using Pkg
Pkg.add(url = "https://github.com/lkdvos/OpSum.jl")
```

## Documentation

See the [documentation](https://lkdvos.github.io/OpSum.jl/dev/) for a worked tutorial, the operator guide, example pages and the API reference.

## Acknowledgements

Inspired by [ITensorMPOConstruction.jl](https://github.com/ITensor/ITensorMPOConstruction.jl) (MIT, © 2024 Ben Corbett and contributors) and the bipartite-graph method of Ren, Li, Jiang and Shuai, [J. Chem. Phys. **153**, 084118 (2020)](https://doi.org/10.1063/5.0018149); OpSum.jl folds non-abelian symmetry into it.
ITensorMPOConstruction.jl asks to be cited as B. Corbett and A. Miyake, [Phys. Rev. B **112**, 165120 (2025)](https://doi.org/10.1103/nzrt-l2j1).

## License

MIT — see [LICENSE](LICENSE).
