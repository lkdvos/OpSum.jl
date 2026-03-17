mutable struct Counter <: Base.Function
    current::Int
end
Counter() = Counter(0)
(x::Counter)() = (x.current += 1; x.current)
