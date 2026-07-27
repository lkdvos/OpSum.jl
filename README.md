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

The compression algorithms are inspired by
[ITensorMPOConstruction.jl](https://github.com/ITensor/ITensorMPOConstruction.jl); the
contribution here is folding non-abelian symmetry into them and making them work on TensorKit
objects. See [Acknowledgements](#acknowledgements) below.

## Installation

This package is not yet registered. It can be installed directly from GitHub:

```julia
julia> using Pkg: Pkg

julia> Pkg.add(url="https://github.com/lkdvos/OpSum.jl")
```

## Documentation

The [documentation](https://lkdvos.github.io/OpSum.jl/dev/) opens with a worked SU(2) Heisenberg
chain, from operator sum to reduced MPO, and continues with example pages covering
[spin chains](https://lkdvos.github.io/OpSum.jl/dev/examples/spin_chains/),
[long-range interactions](https://lkdvos.github.io/OpSum.jl/dev/examples/long_range/),
[ladders and cylinders](https://lkdvos.github.io/OpSum.jl/dev/examples/ladders_and_cylinders/),
[multi-body couplings](https://lkdvos.github.io/OpSum.jl/dev/examples/multibody/) and
[fermions](https://lkdvos.github.io/OpSum.jl/dev/examples/fermions/) — each measuring how the bond
dimension responds. The [reference](https://lkdvos.github.io/OpSum.jl/dev/reference/) documents the
API.

## Acknowledgements

The MPO construction here is inspired by
[ITensorMPOConstruction.jl](https://github.com/ITensor/ITensorMPOConstruction.jl)
(MIT licensed, © 2024 Ben Corbett and contributors), which demonstrated that exact
minimal-bond-dimension MPOs can be built by bipartite-graph compression. OpSum.jl takes those
algorithms and folds in non-abelian symmetry, so that bond indices carry
irreducible-representation labels and the MPO tensors come out as TensorKit `TensorMap`s.

The underlying bipartite-graph / minimum-vertex-cover method is due to J. Ren, W. Li, T. Jiang and
Z. Shuai, [*A general automatic method for optimal construction of matrix product operators using
bipartite graph theory*, J. Chem. Phys. **153**, 084118 (2020)](https://doi.org/10.1063/5.0018149).

ITensorMPOConstruction.jl asks to be cited as B. Corbett and A. Miyake, [*Scaling up the
transcorrelated density matrix renormalization group*, Phys. Rev. B **112**, 165120
(2025)](https://doi.org/10.1103/nzrt-l2j1).

## License

MIT — see [LICENSE](LICENSE).
