# The metric a truncating SVD works in. The SVD backends treat a site's letters as orthonormal, which a
# `Basis` need not satisfy (`n` and `𝟙`, or `X` and `X + Y`, overlap). Lossless results do not care, but a
# truncation would keep the wrong states, so it first rewrites every term over an orthonormalised alphabet,
# compresses, and maps each site entry back.
#
# The inner product is the operator's Hilbert–Schmidt one. For a caterpillar term it factorises:
# `‖t‖² = |coeff|² · Πᵢ dot(aᵢ, aᵢ) / dim(cᵢ)`, times `dot(𝟙, 𝟙)` per idle site; terms with different
# letters or trees are orthogonal once each site's letters are. Dividing each site by `dot(𝟙, 𝟙)` leaves
# the per-charge Gram matrix `G_c[n, m] = dot(eₙ, eₘ) / (dim(c) · dot(𝟙, 𝟙))`, with the identity of norm one.
#
# With `G_c = A'A` (Cholesky, `A` upper triangular), a letter `(c, n)` is `Σ_{m ≤ n} A[m, n] (c, m)` in the
# orthonormal letters and an orthonormal letter `(c, m)` is `Σ_{n ≤ m} A⁻¹[n, m] (c, n)`. The identity is
# element 1 with `A[1, 1] = 1`, so pass-throughs and identity channels are untouched.

using LinearAlgebra: cholesky, Hermitian, UpperTriangular

# Per charge of one site: `(A, A⁻¹)`, or `nothing` when the letters are already orthonormal.
function _lettermetric(B::Basis{I}) where {I}
    T1 = instantiate(passthrough(I), B)
    z = real(dot(T1, T1))
    out = Dictionary{I, NTuple{2, Matrix{ComplexF64}}}()
    trivial = true
    for (c, idx) in pairs(B.bycharge)
        Ts = [instantiate(IrrepOperator(c, n), B) for n in eachindex(idx)]
        G = ComplexF64[dot(a, b) for a in Ts, b in Ts] ./ (dim(c) * z)
        # exact zeros where elements are orthogonal, so `A` keeps their structure
        θ = 1.0e-13 * maximum(abs, G)
        G[abs.(G) .< θ] .= 0
        A = Matrix(cholesky(Hermitian(G)).U)
        trivial &= isapprox(A, one(A); atol = 1.0e-13)
        insert!(out, c, (A, Matrix(inv(UpperTriangular(A)))))
    end
    return trivial ? nothing : out
end

# The terms of `ts` over the orthonormal letters of each site (`metrics[s]` for site `s`). A letter
# can pick up an identity component, which is a pass-through and is dropped like placement drops it.
function _orthonormal_terms(ts::Terms{I}, metrics) where {I}
    out = Term{I}[]
    for t in ts
        choices = map(t.sites, t.keys) do s, key
            m = metrics[s]
            m === nothing && return [(key, one(ComplexF64))]
            A = first(m[key.op.c])
            n = key.op.n
            return [
                (ITOKey{I}(IrrepOperator(key.op.c, k), key.bond, key.vertex), A[k, n])
                    for k in 1:n if !iszero(A[k, n])
            ]
        end
        for pick in Iterators.product(choices...)
            keep = Bool[!ispassthrough(key.op) for (key, _) in pick]
            push!(
                out, Term{I}(
                    t.sites[keep], ITOKey{I}[key for (key, _) in pick][keep],
                    t.coeff * prod(last, pick; init = one(ComplexF64))
                )
            )
        end
    end
    return Terms{I}(out)
end

# One site entry over the orthonormal letters, back in the site's basis.
function _from_orthonormal(op::SiteOperator{I}, metric) where {I}
    letters = IrrepOperator{I}[]
    coeffs = ComplexF64[]
    for (letter, w) in pairs(op)
        Ainv = last(metric[letter.c])
        m = letter.n
        for n in 1:m
            x = Ainv[n, m] * w
            iszero(x) && continue
            l = IrrepOperator(letter.c, n)
            j = findfirst(==(l), letters)
            j === nothing ? (push!(letters, l); push!(coeffs, x)) : (coeffs[j] += x)
        end
    end
    return SiteOperator{I}(letters, coeffs)
end

# `_irrep_channels` for a truncating SVD on a lattice of bases: compress in the orthonormal letters,
# then rewrite every site's entries in that site's basis.
function _compress_in_metric(tt_terms::Terms{I}, lat::FiniteChain, strategy) where {I}
    N = length(lat)
    bymetric = IdDict{Any, Any}()
    metrics = [get!(() -> _lettermetric(lat[s]), bymetric, lat[s]) for s in 1:N]
    all(isnothing, metrics) && return nothing
    tt = ITOTermTable(_orthonormal_terms(tt_terms, metrics), N)
    Ws, secs, start, finish = _irrep_channels(tt, nvertices(tt), strategy)
    for (s, W) in enumerate(Ws)
        metrics[s] === nothing && continue
        W.nzval .= _from_orthonormal.(W.nzval, Ref(metrics[s]))
    end
    return Ws, secs, start, finish
end
