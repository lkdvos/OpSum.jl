# On-site products of letters as pure coefficient arithmetic: the structure constants `e_x · e_y = Σ_z N[x,y]_z e_z`
# of a `Basis` are computed once per pair from the tensors (the written-order block `_written_block`, projected
# back by the single-site route) and memoised; products of `SiteOperator`s then never touch a tensor.
#
# The two charge legs are fused to the unique channel `x.c ⊗ y.c`, with the graded sign `a[i] * b[i]` uses
# (`c†` then `c` is `n`). Abelian symmetries only: the channel must be forced and the legs reorder freely.

using TensorKit: Sector, FusionStyle, BraidingStyle

function _checkkron(::Type{I}) where {I <: Sector}
    _canreorder(I) || throw(
        ArgumentError(
            "on-site products of letters need an abelian symmetry (unique fusion, symmetric braiding), " *
                "got $I with $(FusionStyle(I)) fusion and $(BraidingStyle(I)) braiding"
        )
    )
    return nothing
end

# One table per basis, found by identity: hashing a `Basis` hashes its tensors, far too slow for a lookup
# in the inner loop of a product. The table keeps its basis alive, so an identity key cannot be recycled.
const _LETTER_PRODUCT_CACHE = IdDict{Any, Any}()

function _product_table(B::Basis{I}) where {I}
    return @lock OPCACHE_LOCK get!(_LETTER_PRODUCT_CACHE, B) do
        Dict{Tuple{Letter{I}, Letter{I}}, SiteOperator{I}}()
    end
end

# `e_x · e_y` (`x` written first, leftmost) in the basis `B`: a `SiteOperator` on letters of charge `x.c ⊗ y.c`.
function _letter_product(B::Basis{I}, x::Letter{I}, y::Letter{I}) where {I}
    ispassthrough(x) && return SiteOperator(y)
    ispassthrough(y) && return SiteOperator(x)
    return cached(_product_table(B), (x, y)) do
        _checkkron(I)                    # once per pair, not per product
        tot = only(x.c ⊗ y.c)
        keys = Factor{I}[Factor{I}(x, x.c), Factor{I}(y, tot)]
        h = _written_block(keys, [1, 1], [B])
        coeff = (_isfermionodd(x.c) && _isfermionodd(y.c)) ? -1.0 : 1.0
        return project(h, B) * coeff
    end
end

# Bilinear extension of `_letter_product` to `SiteOperator`s (letters of any charges).
function _mul(B::Basis{I}, X::SiteOperator{I}, Y::SiteOperator{I}) where {I}
    isone(X) && return Y
    isone(Y) && return X
    acc = Dict{Letter{I}, ComplexF64}()
    for (x, cx) in pairs(X), (y, cy) in pairs(Y)
        for (z, cz) in pairs(_letter_product(B, x, y))
            acc[z] = get(acc, z, zero(ComplexF64)) + cx * cy * cz
        end
    end
    letters = sort!(collect(keys(acc)))
    coeffs = ComplexF64[acc[l] for l in letters]
    keep = abs.(coeffs) .> 1.0e-14 * max(1.0, maximum(abs, coeffs; init = 0.0))
    return SiteOperator{I}(letters[keep], coeffs[keep])
end
