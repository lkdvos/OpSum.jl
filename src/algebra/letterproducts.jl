# On-site products of letters as pure coefficient arithmetic: the structure constants `e_x · e_y = Σ_z N[x,y]_z e_z`
# of a `Basis` are computed once per pair from the tensors (the written-order block `written_block`, projected
# back by the single-site route) and memoised in `B.products`; products of `SiteOperator`s then never touch
# a tensor.
#
# The two charge legs are fused to the unique channel `x.c ⊗ y.c`, with the graded sign `a[i] * b[i]` uses
# (`c†` then `c` is `n`). Abelian symmetries only: the channel must be forced and the legs reorder freely.

using TensorKit: Sector, FusionStyle, BraidingStyle

function checkkron(::Type{I}) where {I <: Sector}
    canreorder(I) || throw(
        ArgumentError(
            "on-site products of letters need an abelian symmetry (unique fusion, symmetric braiding), " *
                "got $I with $(FusionStyle(I)) fusion and $(BraidingStyle(I)) braiding"
        )
    )
    return nothing
end

# `e_x · e_y` (`x` written first, leftmost) in the basis `B`: a `SiteOperator` on letters of charge `x.c ⊗ y.c`.
function letter_product(B::Basis{I}, x::Letter{I}, y::Letter{I}) where {I}
    ispassthrough(x) && return SiteOperator(y)
    ispassthrough(y) && return SiteOperator(x)
    return cached(B.products, (x, y)) do
        checkkron(I)                    # once per pair, not per product
        tot = only(x.c ⊗ y.c)
        keys = Factor{I}[Factor{I}(x, x.c), Factor{I}(y, tot)]
        h = written_block(keys, [1, 1], [B])
        coeff = (isfermionodd(x.c) && isfermionodd(y.c)) ? -1.0 : 1.0
        return project(h, B) * coeff
    end
end

# Bilinear extension of `letter_product` to `SiteOperator`s (letters of any charges).
function site_product(B::Basis{I}, X::SiteOperator{I}, Y::SiteOperator{I}) where {I}
    isone(X) && return Y
    isone(Y) && return X
    acc = Dict{Letter{I}, ComplexF64}()
    for (x, cx) in pairs(X), (y, cy) in pairs(Y)
        for (z, cz) in pairs(letter_product(B, x, y))
            acc[z] = get(acc, z, zero(ComplexF64)) + cx * cy * cz
        end
    end
    letters = sort!(collect(keys(acc)))
    coeffs = ComplexF64[acc[l] for l in letters]
    keep = abs.(coeffs) .> 1.0e-14 * max(1.0, maximum(abs, coeffs; init = 0.0))
    return SiteOperator{I}(letters[keep], coeffs[keep])
end
