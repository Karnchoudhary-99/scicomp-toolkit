# Eigenvalue estimation: power iteration and inverse iteration.

"""
    rayleigh_quotient(A, v)

The scalar `λ` minimizing `‖A*v - λ*v‖` for a non-zero vector `v`.
"""
function rayleigh_quotient(A::AbstractMatrix, v::AbstractVector)
    return dot_product(v, A * v) / dot_product(v, v)
end

"""
    power_iteration(A; maxiter=1000, tol=1e-10)

Dominant eigenpair of `A` by power iteration. Returns
`(λ, v, iterations, converged)` with `v` a unit vector.
"""
function power_iteration(A::AbstractMatrix; maxiter::Integer = 1000, tol::Real = 1e-10)
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("power iteration needs a square matrix"))
    v = normalize_vector(ones(Float64, n))
    λ = rayleigh_quotient(A, v)
    for iter in 1:maxiter
        w = A * v
        nw = norm_p(w)
        nw == 0 && throw(ArgumentError("power iteration hit the zero vector"))
        v = w ./ nw
        λnew = rayleigh_quotient(A, v)
        abs(λnew - λ) < tol && return λnew, v, iter, true
        λ = λnew
    end
    return λ, v, maxiter, false
end

"""
    inverse_iteration(A, shift; maxiter=1000, tol=1e-10)

Eigenpair of `A` nearest to `shift`, by inverse iteration on `A - shift*I`.
Returns `(λ, v, iterations, converged)`.
"""
function inverse_iteration(A::AbstractMatrix, shift::Real; maxiter::Integer = 1000, tol::Real = 1e-10)
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("inverse iteration needs a square matrix"))
    M = A .- shift .* Matrix{Float64}(I, n, n)
    v = normalize_vector(ones(Float64, n))
    λ = rayleigh_quotient(A, v)
    for iter in 1:maxiter
        w = gauss_solve(M, v)
        v = normalize_vector(w)
        λnew = rayleigh_quotient(A, v)
        abs(λnew - λ) < tol && return λnew, v, iter, true
        λ = λnew
    end
    return λ, v, maxiter, false
end
