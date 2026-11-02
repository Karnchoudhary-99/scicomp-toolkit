# Scalar root finding: bisection, Newton, secant.

"""
    RootResult

Result of a root finder: `root`, `iterations`, and `converged`.
"""
struct RootResult
    root::Float64
    iterations::Int
    converged::Bool
end

"""
    bisection(f, a, b; tol=1e-10, maxiter=1000)

Bisection on a sign-changing bracket `[a, b]`. Throws `ArgumentError`
when `f(a)` and `f(b)` share a sign.
"""
function bisection(f::Function, a::Real, b::Real; tol::Real = 1e-10, maxiter::Integer = 1000)
    fa, fb = f(a), f(b)
    fa == 0 && return RootResult(a, 0, true)
    fb == 0 && return RootResult(b, 0, true)
    fa * fb > 0 && throw(ArgumentError("bisection needs f(a) and f(b) of opposite sign"))
    for iter in 1:maxiter
        m = (a + b) / 2
        fm = f(m)
        (fm == 0 || (b - a) / 2 < tol) && return RootResult(m, iter, true)
        fa * fm < 0 ? (b = m) : (a = m; fa = fm)
    end
    return RootResult((a + b) / 2, maxiter, false)
end

"""
    newton_root(f, df, x0; tol=1e-10, maxiter=1000)

Newton's method from `x0` using derivative `df`. Throws `ArgumentError`
on a zero derivative step.
"""
function newton_root(f::Function, df::Function, x0::Real; tol::Real = 1e-10, maxiter::Integer = 1000)
    x = float(x0)
    for iter in 1:maxiter
        d = df(x)
        d == 0 && throw(ArgumentError("Newton's method hit a zero derivative"))
        xnew = x - f(x) / d
        abs(xnew - x) < tol && return RootResult(xnew, iter, true)
        x = xnew
    end
    return RootResult(x, maxiter, false)
end

"""
    secant_root(f, x0, x1; tol=1e-10, maxiter=1000)

Secant method from two starting points (no derivative needed).
"""
function secant_root(f::Function, x0::Real, x1::Real; tol::Real = 1e-10, maxiter::Integer = 1000)
    x_prev, x = float(x0), float(x1)
    f_prev, fx = f(x_prev), f(x)
    for iter in 1:maxiter
        denom = fx - f_prev
        denom == 0 && throw(ArgumentError("secant method hit a zero denominator"))
        xnew = x - fx * (x - x_prev) / denom
        abs(xnew - x) < tol && return RootResult(xnew, iter, true)
        x_prev, f_prev, x, fx = x, fx, xnew, f(xnew)
    end
    return RootResult(x, maxiter, false)
end
