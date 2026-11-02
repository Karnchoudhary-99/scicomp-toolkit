# Linear solvers: direct (LU/Cholesky) and iterative (Jacobi, Gauss-Seidel).

"""
    forward_substitution(L, b)

Solve `L * x == b` for lower-triangular `L` with non-zero diagonal.
"""
function forward_substitution(L::AbstractMatrix, b::AbstractVector)
    n = length(b)
    x = zeros(Float64, n)
    for i in 1:n
        x[i] = (b[i] - sum(L[i, j] * x[j] for j in 1:(i - 1); init = 0.0)) / L[i, i]
    end
    return x
end

"""
    back_substitution(U, b)

Solve `U * x == b` for upper-triangular `U` with non-zero diagonal.
"""
function back_substitution(U::AbstractMatrix, b::AbstractVector)
    n = length(b)
    x = zeros(Float64, n)
    for i in n:-1:1
        x[i] = (b[i] - sum(U[i, j] * x[j] for j in (i + 1):n; init = 0.0)) / U[i, i]
    end
    return x
end

"""
    gauss_solve(A, b)

Solve `A * x == b` by Gaussian elimination with partial pivoting
(via [`lu_factor`](@ref)). Throws `SingularException` for singular `A`.
"""
function gauss_solve(A::AbstractMatrix, b::AbstractVector)
    F = lu_factor(A)
    pb = b[F.p]
    return back_substitution(F.U, forward_substitution(F.L, pb))
end

"""
    cholesky_solve(A, b)

Solve `A * x == b` for symmetric positive-definite `A`.
"""
function cholesky_solve(A::AbstractMatrix, b::AbstractVector)
    L = cholesky_factor(A)
    return back_substitution(Matrix(L'), forward_substitution(L, b))
end

"""
    jacobi_solve(A, b; maxiter=1000, tol=1e-10)

Iterative Jacobi solver. Returns `(x, iterations, converged)`.
Throws `ArgumentError` on a zero diagonal entry.
"""
function jacobi_solve(A::AbstractMatrix, b::AbstractVector; maxiter::Integer = 1000, tol::Real = 1e-10)
    n = length(b)
    any(A[i, i] == 0 for i in 1:n) && throw(ArgumentError("Jacobi needs a non-zero diagonal"))
    x = zeros(Float64, n)
    for iter in 1:maxiter
        xnew = [(b[i] - sum(A[i, j] * x[j] for j in 1:n if j != i; init = 0.0)) / A[i, i] for i in 1:n]
        norm_p(xnew .- x) < tol && return xnew, iter, true
        x = xnew
    end
    return x, maxiter, false
end

"""
    gauss_seidel_solve(A, b; maxiter=1000, tol=1e-10)

Iterative Gauss-Seidel solver (in-place updates, usually faster than Jacobi).
Returns `(x, iterations, converged)`.
"""
function gauss_seidel_solve(A::AbstractMatrix, b::AbstractVector; maxiter::Integer = 1000, tol::Real = 1e-10)
    n = length(b)
    any(A[i, i] == 0 for i in 1:n) && throw(ArgumentError("Gauss-Seidel needs a non-zero diagonal"))
    x = zeros(Float64, n)
    for iter in 1:maxiter
        xold = copy(x)
        for i in 1:n
            x[i] = (b[i] - sum(A[i, j] * x[j] for j in 1:n if j != i; init = 0.0)) / A[i, i]
        end
        norm_p(x .- xold) < tol && return x, iter, true
    end
    return x, maxiter, false
end
