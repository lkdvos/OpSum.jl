# Approach:
# Symbolic operations do not simplify by default, and all simplifications are done through
# the `simplify` function. This is the most general, but might be slow, so needs to be 
# evaluated.

"""
    SymbolicOperator{I,S,C<:Number}

This module defines the `SymbolicOperator` type, which is used to represent a symbolic operator algebra.
The elementary operators are defined through labels of type `I`, which can be any type.
These operators act on sites of type `S`, which can also be any type.
The coefficients of the operators are of type `C`, which is restricted to be a subtype of `Number`.

Finally, the different flavours of the types in this module ensure that these operators
function as a (symbolic) algebra over `C`, with scalar multiplication, addition, and multiplication.
"""
@data SymbolicOperator{I,S,C<:Number} begin
    # Single operator
    struct Operator
        flavour::I
        site::S
    end

    # Scaled operator
    struct Scaled
        scalar::C
        operator::SymbolicOperator{I,S,C}
    end

    # Addition: commutative -> Set-like
    struct Sum
        arguments::Dictionary{SymbolicOperator{I,S,C},C}
    end

    # Multiplication -> non-commutative -> List-like
    # TODO: SymbolicUtils has: x^m * y^n * ... as [x => m, y => n, ...]
    struct Product
        arguments::Vector{SymbolicOperator{I,S,C}}
    end
end

import .SymbolicOperator: Operator, Scaled, Sum, Product
const AbstractOperator{I,S,C} = SymbolicOperator.Type{I,S,C}
export AbstractOperator

# Constructors
# ------------
# TODO: default scalartype?
function Operator(flavour::I, site::S) where {I,S}
    return Operator{I,S,Bool}(flavour, site)
end

macro op_str(op)
    op_split = split(op, "_")
    @assert length(op_split) == 2 "Expected `flavour_site`, got \"$op\""
    flavour = QuoteNode(Symbol(op_split[1]))
    site = something(tryparse(Int, op_split[2]), Symbol(op_split[2]))
    return esc(:(Operator($flavour, $site)))
end

# Converters
# ----------
function Base.convert(
    ::Type{AbstractOperator{I,S,C}}, O::AbstractOperator{I,S}
) where {I,S,C}
    scalartype(O) === C && return O
    return @match O begin
        Operator(flavour, site) => Operator{I,S,C}(flavour, site)
        Scaled(scalar, operator) =>
            Scaled{I,S,C}(convert(C, scalar), convert(AbstractOperator{I,S,C}, operator))
        Sum(args) => Sum{I,S,C}(
            Dictionary(
                map(Base.Fix1(convert, AbstractOperator{I,S,C}), keys(args)),
                map(Base.Fix1(convert, C), args),
            ),
        )
        Product(arguments) =>
            Product{I,S,C}(map(Base.Fix1(convert, AbstractOperator{I,S,C}), arguments))
    end
end

# Promotions
function Base.promote_rule(
    ::Type{AbstractOperator{I,S,C₁}}, ::Type{AbstractOperator{I,S,C₂}}
) where {I,S,C₁,C₂}
    return AbstractOperator{I,S,promote_type(C₁, C₂)}
end

# Utility
# -------
scalartype(::Type{AbstractOperator{I,S,C}}) where {I,S,C} = C
scalartype(O::AbstractOperator) = scalartype(typeof(O))

flavourtype(::Type{AbstractOperator{I,S,C}}) where {I,S,C} = I
flavourtype(O::AbstractOperator) = flavourtype(typeof(O))

sitetype(::Type{AbstractOperator{I,S,C}}) where {I,S,C} = S
sitetype(O::AbstractOperator) = sitetype(typeof(O))

# Sorting
# -------
# attempt to define a total order on operators, both for printing and easily comparing
# sites -> lexicographic order
# operator type -> (Operator & Scaled) < Product < Sum?
function Base.isless(O₁::AbstractOperator{I,S}, O₂::AbstractOperator{I,S}) where {I,S}
    return @match O₁ begin
        Operator(flavour₁, site₁) => @match O₂ begin
            Operator(flavour₂, site₂) => (site₁, flavour₁) < (site₂, flavour₂)
            Scaled(λ₂, o₂) => O₁ ≤ o₂ # non-scaled goes first
            Product(args₂) => [O₁] < args₂
            Sum(args₂) => true
        end

        Scaled(λ₁, o₁) => @match O₂ begin
            Operator(_, _) => !(O₁ ≥ O₂)
            Scaled(λ₂, o₂) => (o₁, λ₁) < (o₂, λ₂)
            Product(args₂) => [O₁] < args₂
            Sum(args₂) => true
        end

        Product(args₁) => @match O₂ begin
            Operator(_, _) => !(O₁ ≥ O₂)
            Scaled(_, _) => !(O₁ ≥ O₂)
            Product(args₂) => args₁ < args₂
            Sum(_) => true
        end

        Sum(args₁) => @match O2 begin
            Operator(_, _) => !(O₁ ≥ O₂)
            Scaled(_, _) => !(O₁ ≥ O₂)
            Product(_) => !(O₁ ≥ O₂)
            Sum(args₂) => args₁ < args₂
        end
    end
end

# Algebra operations
# ------------------
# These are defined to keep the algebra closed under the operations
# of scalar multiplication, addition, and multiplication.
# Mostly, they are implemented lazily, but we make some attempts to simplify nesting
import Base: +, *, -, /, \, one, zero

# Scalar multiplication
function (λ::C * O::AbstractOperator{I,S,C}) where {I,S,C<:Number}
    isone(λ) && return O
    return @match O begin
        Sum(arguments) => Sum(map(Base.Fix1(*, λ), arguments))
        Scaled(scalar, operator) => Scaled(λ * scalar, operator)
        x => Scaled(λ, x)
    end
end
function (λ::Number * O::AbstractOperator)
    T = Base.promote_op(*, typeof(λ), scalartype(O))
    return T(λ) * convert(AbstractOperator{flavourtype(O),sitetype(O),T}, O)
end

(O::AbstractOperator * λ::Number) = λ * O
(O::AbstractOperator / λ::Number) = O * inv(λ)
(λ::Number \ O::AbstractOperator) = inv(λ) * O

# Addition
function (O₁::SO + O₂::SO) where {I,S,C,SO<:AbstractOperator{I,S,C}}
    return @match (O₁, O₂) begin
        (Sum(args₁), Sum(args₂)) => Sum(mergewith(+, args₁, args₂))
        (Sum(args₁), Scaled(λ₂, o₂)) => Sum(setwith!(+, copy(args₁), o₂, λ₂))
        (Sum(args₁), o₂) => Sum(setwith!(+, copy(args₁), o₂, one(C)))
        (o1, Sum(_)) => O₂ + O₁

        (Product(args₁), Product(args₂)) => Sum(Dictionary([O₁, O₂], ones(C, 2)))
        (Product(args₁), Scaled(λ₂, o₂)) => Sum(Dictionary([O₁, o₂], [one(C), λ₂]))
        (Product(args₁), o₂) => Sum(Dictionary([O₁, o₂], ones(C, 2)))
        (o1, Product(_)) => O₂ + O₁

        (Operator(_, _), Operator(_, _)) =>
            O₁ == O₂ ? 2 * O₁ : Sum(Dictionary([O₁, O₂], ones(C, 2)))
        (Operator(_, _), Scaled(λ₂, o₂)) =>
            O₁ == o₂ ? (λ₂ + 1) * O₁ : Sum(Dictionary([O₁, o₂], [one(C), λ₂]))
        (Scaled(λ₁, o₁), Operator(_, _)) =>
            o₁ == O₂ ? (λ₁ + 1) * o₁ : Sum(Dictionary([o₁, O₂], [λ₁, one(C)]))
        (Scaled(λ₁, o₁), Scaled(λ₂, o₂)) =>
            o₁ == o₂ ? (λ₁ + λ₂) * o₁ : Sum(Dictionary([o₁, o₂], [λ₁, λ₂]))

        (_, _) => throw(ArgumentError("Cannot add $O₁ and $O₂"))
    end
end
(O1::AbstractOperator + O2::AbstractOperator) = +(promote(O1, O2)...)

# Subtraction
-(O::AbstractOperator) = -one(scalartype(O)) * O
(O1::SO - O2::SO) where {SO<:AbstractOperator} = O1 + (-O2)
(O1::AbstractOperator - O2::AbstractOperator) = -(promote(O1, O2)...)

# Operator multiplication
function (O₁::SO * O₂::SO) where {I,S,C,SO<:AbstractOperator{I,S,C}}
    return @match O₁, O₂ begin
        (Scaled(λ₁, o₁), o₂) => λ * (o₁ * o₂)
        (o₁, Scaled(λ₂, o₂)) => λ * (o₁ * o₂)
        (o₁, o₂) => Product([o₁, o₂])
    end
end

# show
# ----
import Base: show, show_unquoted

# show (parseable)
function show(io::IO, O::AbstractOperator)
    @match O begin
        # Operator => op"flavour_site"
        Operator(flavour, site) => begin
            if !get(io, :_print_op, true)::Bool
                print(io, flavour, '_', site)
            else
                print(io, "op\"", flavour, '_', site, "\"")
            end
        end

        # Scaled => scalar * operator
        Scaled(scalar, operator) => begin
            if isone(scalar)
                show(io, operator)
            elseif isreal(scalar) && isone(abs(scalar))
                print(io, "-")
                show(io, operator)
            else
                show_unquoted(io, scalar, 0, Base.operator_precedence(:*))
                print(io, " * ")
                show_unquoted(io, operator, 0, Base.operator_precedence(:*))
            end
        end

        # Sum => λ₁ * o₁ + λ₂ * o₂ + ...
        Sum(arguments) => begin
            precedence = Base.operator_precedence(:+)
            for (i, (operator, scalar)) in enumerate(pairs(sortkeys(arguments)))
                if i == 1
                    show_unquoted(io, operator * scalar, 0, precedence)
                else
                    if isreal(scalar) && real(scalar) < 0
                        print(io, " - ")
                        scalar = abs(scalar)
                    elseif scalar isa Complex
                        re, im = real(scalar), imag(scalar)
                        if re < 0
                            print(io, " - ")
                            scalar = complex(abs(re), im)
                        else
                            print(io, " + ")
                        end
                    else # not real nor complex so just print
                        print(io, " + ")
                    end
                    show_unquoted(io, operator * scalar, 0, precedence)
                end
            end
        end

        # Product => o₁ * o₂ * ...
        Product(arguments) => begin
            precedence = Base.operator_precedence(:*)
            for (i, operator) in enumerate(arguments)
                if i == 1
                    show_unquoted(io, operator, 0, precedence)
                else
                    print(io, " * ")
                    show_unquoted(io, operator, 0, precedence)
                end
            end
        end
    end
end

# show (human-readable)
function show(io::IO, ::MIME"text/plain", O::AbstractOperator)
    ioctx = IOContext(io, :_print_op => false)
    return show(ioctx, O)
end

# show in the context of an expression
function show_unquoted(io::IO, O::AbstractOperator, indent::Int, precedence::Int)
    @match O begin
        # operators are always fine
        Operator(flavour, site) => show(io, O)

        # scaled operators might be (scalar * operator)
        Scaled(scalar, operator) => begin
            if isone(scalar) ||
                (isreal(scalar) && isone(abs(scalar))) ||
                Base.operator_precedence(:*) > precedence
                show(io, O)
            else
                print(io, "(")
                show(io, O)
                print(io, ")")
            end
        end

        # sums might be (o1 + o2 + ...)
        Sum(arguments) => begin
            if length(arguments) > 1 && Base.operator_precedence(:+) ≤ precedence
                print(io, "(")
                show(io, O)
                print(io, ")")
            else
                show(io, O)
            end
        end

        # products might be (o1 * o2 * ...)
        Product(arguments) => begin
            if length(arguments) > 1 && Base.operator_precedence(:*) ≤ precedence
                print(io, "(")
                show(io, O)
                print(io, ")")
            else
                show(io, O)
            end
        end
    end
end
