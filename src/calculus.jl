# Numerical differentiation and integration.

"""
    central_difference(f, x; h=1e-5)

First derivative of scalar `f` at `x` by the central difference
`(f(x+h) - f(x-h)) / (2h)`.
"""
central_difference(f::Function, x::Real; h::Real = 1e-5) = (f(x + h) - f(x - h)) / (2h)

"""
    second_difference(f, x; h=1e-4)

Second derivative of scalar `f` at `x` by
`(f(x+h) - 2f(x) + f(x-h)) / h^2`.
"""
second_difference(f::Function, x::Real; h::Real = 1e-4) = (f(x + h) - 2f(x) + f(x - h)) / h^2

"""
    gradient_central(f, x; h=1e-5)

Gradient of scalar-valued `f` at vector `x` by central differences.
"""
function gradient_central(f::Function, x::AbstractVector; h::Real = 1e-5)
    g = zeros(Float64, length(x))
    for i in eachindex(x)
        xp, xm = copy(x), copy(x)
        xp[i] += h
        xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2h)
    end
    return g
end

"""
    trapezoid(f, a, b, n)

Integral of `f` over `[a, b]` by the composite trapezoid rule with `n`
subintervals.
"""
function trapezoid(f::Function, a::Real, b::Real, n::Integer)
    n >= 1 || throw(ArgumentError("n must be positive"))
    h = (b - a) / n
    s = (f(a) + f(b)) / 2
    for i in 1:(n - 1)
        s += f(a + i * h)
    end
    return s * h
end

"""
    simpson(f, a, b, n)

Integral of `f` over `[a, b]` by composite Simpson's rule with `n`
subintervals (`n` must be even).
"""
function simpson(f::Function, a::Real, b::Real, n::Integer)
    n >= 2 && iseven(n) || throw(ArgumentError("Simpson's rule needs an even n >= 2"))
    h = (b - a) / n
    s = f(a) + f(b)
    for i in 1:(n - 1)
        s += (isodd(i) ? 4 : 2) * f(a + i * h)
    end
    return s * h / 3
end
