# Unconstrained minimization: gradient descent and Newton's method.

"""
    OptimizeResult

Result of an optimizer: `x` (minimizer), `fmin`, `iterations`, `converged`.
"""
struct OptimizeResult{T}
    x::T
    fmin::Float64
    iterations::Int
    converged::Bool
end

"""
    gradient_descent(f, grad, x0; lr=0.1, maxiter=1000, tol=1e-8)

Fixed-step gradient descent on scalar-valued `f` with gradient `grad`.
Works for scalar or vector `x0`.
"""
function gradient_descent(f::Function, grad::Function, x0; lr::Real = 0.1, maxiter::Integer = 1000, tol::Real = 1e-8)
    x = x0
    for iter in 1:maxiter
        g = grad(x)
        step = lr .* g
        xnew = x .- step
        step_norm = step isa AbstractVector ? norm_p(step) : abs(step)
        step_norm < tol && return OptimizeResult(xnew, f(xnew), iter, true)
        x = xnew
    end
    return OptimizeResult(x, f(x), maxiter, false)
end

"""
    newton_minimize(f, grad, hess, x0; maxiter=100, tol=1e-10)

Newton's method for scalar minimization: `x -= grad / hess`.
Throws `ArgumentError` on a zero second derivative.
"""
function newton_minimize(f::Function, grad::Function, hess::Function, x0::Real; maxiter::Integer = 100, tol::Real = 1e-10)
    x = float(x0)
    for iter in 1:maxiter
        h = hess(x)
        h == 0 && throw(ArgumentError("Newton minimization hit a zero second derivative"))
        xnew = x - grad(x) / h
        abs(xnew - x) < tol && return OptimizeResult(xnew, f(xnew), iter, true)
        x = xnew
    end
    return OptimizeResult(x, f(x), maxiter, false)
end
