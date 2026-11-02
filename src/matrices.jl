# Dense-matrix helpers and classic test matrices.

"""
    frobenius_norm(A)

Square root of the sum of squared entries of `A`.
"""
frobenius_norm(A::AbstractMatrix) = sqrt(sum(abs2, A))

"""
    matrix_trace(A)

Sum of the diagonal of a square matrix.
"""
function matrix_trace(A::AbstractMatrix)
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("trace needs a square matrix"))
    return sum(A[i, i] for i in axes(A, 1))
end

"""
    is_symmetric(A; atol=1e-8)

True when `A` equals its transpose within `atol`.
"""
function is_symmetric(A::AbstractMatrix; atol::Real = 1e-8)
    size(A, 1) == size(A, 2) || return false
    return all(abs.(A .- A') .<= atol)
end

"""
    diag_vector(A)

Diagonal of `A` as a vector.
"""
diag_vector(A::AbstractMatrix) = [A[i, i] for i in 1:min(size(A)...)]


"""
    diag_matrix(v)

Diagonal matrix with `v` on the diagonal.
"""
function diag_matrix(v::AbstractVector)
    n = length(v)
    D = zeros(promote_type(eltype(v), Float64), n, n)
    for i in 1:n
        D[i, i] = v[i]
    end
    return D
end

"""
    hilbert_matrix(n)

The `n×n` Hilbert matrix `H[i, j] = 1 / (i + j - 1)`, a classic
ill-conditioned test case for linear solvers.
"""
function hilbert_matrix(n::Integer)
    n >= 1 || throw(ArgumentError("n must be positive"))
    return [1.0 / (i + j - 1) for i in 1:n, j in 1:n]
end

"""
    vandermonde(x)

Vandermonde matrix with rows `[1, x[i], x[i]^2, …, x[i]^(n-1)]`.
"""
function vandermonde(x::AbstractVector)
    n = length(x)
    return [x[i]^(j - 1) for i in 1:n, j in 1:n]
end
