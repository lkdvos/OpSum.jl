# Metric of a truncating SVD. The SVD backends treat a site's letters as orthonormal, which a `Basis`
# need not satisfy, so terms are rewritten over per-charge orthonormalised letters, compressed, and
# each site entry mapped back. The Hilbert–Schmidt norm of a caterpillar term factorises per site, so the
# per-charge Gram matrix is `G_c[n, m] = dot(eₙ, eₘ) / (dim(c) · dot(𝟙, 𝟙))` (`lettermetric`, stored in
# the `Basis`). With `G_c = A'A` (Cholesky), letter `(c, n)` is `Σ_{m ≤ n} A[m, n] (c, m)` and the
# orthonormal `(c, m)` is `Σ_{n ≤ m} A⁻¹[n, m] (c, n)`; the identity is element 1 with `A[1, 1] = 1`, so
# pass-throughs are untouched.

using LinearAlgebra: cholesky, Hermitian, UpperTriangular

# Per charge of one site: `(A, A⁻¹)`, or `nothing` when the letters are already orthonormal. Computed from
# the per-charge Gram matrices when a `Basis` is built, and stored in it (`B.metric`).
function lettermetric(grams::Dictionary{I, Matrix{ComplexF64}}) where {I}
    z = real(grams[unit(I)][1, 1])    # ⟨𝟙, 𝟙⟩, the identity being element 1
    out = Dictionary{I, NTuple{2, Matrix{ComplexF64}}}()
    trivial = true
    for (c, Gc) in pairs(grams)
        G = Gc ./ (dim(c) * z)
        # exact zeros where elements are orthogonal, so `A` keeps their structure
        θ = 1.0e-13 * maximum(abs, G)
        G[abs.(G) .< θ] .= 0
        A = Matrix(cholesky(Hermitian(G)).U)
        trivial &= isapprox(A, one(A); atol = 1.0e-13)
        insert!(out, c, (A, Matrix(inv(UpperTriangular(A)))))
    end
    return trivial ? nothing : out
end

# `ts` over the orthonormal letters (`metrics[s]` for site `s`); identity components are dropped.
function orthonormal_terms(ts::Terms{I}, metrics) where {I}
    out = Term{I}[]
    for t in ts
        choices = map(t.sites, t.factors) do s, key
            m = metrics[s]
            m === nothing && return [(key, one(ComplexF64))]
            A = first(m[key.letter.c])
            n = key.letter.n
            return [
                (Factor{I}(Letter(key.letter.c, k), key.bond), A[k, n])
                    for k in 1:n if !iszero(A[k, n])
            ]
        end
        for pick in Iterators.product(choices...)
            keep = Bool[!ispassthrough(key.letter) for (key, _) in pick]
            push!(
                out, Term{I}(
                    t.sites[keep], Factor{I}[key for (key, _) in pick][keep],
                    t.coeff * prod(last, pick; init = one(ComplexF64))
                )
            )
        end
    end
    return Terms{I}(out)
end

# One site entry over the orthonormal letters, back in the site's basis.
function _from_orthonormal(op::SiteOperator{I}, metric) where {I}
    letters = Letter{I}[]
    coeffs = ComplexF64[]
    for (letter, w) in pairs(op)
        Ainv = last(metric[letter.c])
        m = letter.n
        for n in 1:m
            x = Ainv[n, m] * w
            iszero(x) && continue
            l = Letter(letter.c, n)
            j = findfirst(==(l), letters)
            j === nothing ? (push!(letters, l); push!(coeffs, x)) : (coeffs[j] += x)
        end
    end
    return SiteOperator{I}(letters, coeffs)
end

# `_sweep_channels` in the orthonormal letters, entries mapped back to each site's basis; `nothing`
# when every site is already orthonormal.
function compress_in_metric(tt_terms::Terms{I}, lat::FiniteChain, strategy) where {I}
    N = length(lat)
    metrics = [lat[s].metric for s in 1:N]
    all(isnothing, metrics) && return nothing
    tt = TermTable(orthonormal_terms(tt_terms, metrics), N)
    Ws, secs, start, finish = _sweep_channels(tt, nvertices(tt), strategy)
    for (s, W) in enumerate(Ws)
        metrics[s] === nothing && continue
        W.nzval .= _from_orthonormal.(W.nzval, Ref(metrics[s]))
    end
    return Ws, secs, start, finish
end
