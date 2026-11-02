# Matrix decompositions: LU with partial pivoting, Cholesky, QR (Gram-Schmidt).

"""
    LUFactor

Result of [`lu_factor`](@ref): `L` unit lower-triangular, `U` upper-triangular,
and permutation vector `p` such that `A[p, :] == L * U`.
"""
struct LUFactor{T}
    L::Matrix{T}
    U::Matrix{T}
    p::Vector{Int}
end

"""
    lu_factor(A)

LU decomposition with partial pivoting. Throws `SingularException` on a
numerically zero pivot.
"""
function lu_factor(A::AbstractMatrix)
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("LU needs a square matrix"))
    U = float.(copy(A))
    L = Matrix{Float64}(I, n, n)
    p = collect(1:n)
    for k in 1:(n - 1)
        # Partial pivot: largest entry in column k at or below row k.
        pivot = argmax(abs.(U[k:n, k])) + k - 1
        abs(U[pivot, k]) == 0 && throw(SingularException(k))
        if pivot != k
            U[k, :], U[pivot, :] = U[pivot, :], U[k, :]
            L[k, 1:(k - 1)], L[pivot, 1:(k - 1)] = L[pivot, 1:(k - 1)], L[k, 1:(k - 1)]
            p[k], p[pivot] = p[pivot], p[k]
        end
        for i in (k + 1):n
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:n] .-= L[i, k] .* U[k, k:n]
        end
    end
    abs(U[n, n]) == 0 && throw(SingularException(n))
    return LUFactor(L, U, p)
end

"""
    cholesky_factor(A)

Lower-triangular `L` with `A == L * L'` for a symmetric positive-definite
matrix. Throws `PosDefException` otherwise.
"""
function cholesky_factor(A::AbstractMatrix)
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("Cholesky needs a square matrix"))
    is_symmetric(A) || throw(PosDefException(0))
    L = zeros(Float64, n, n)
    for i in 1:n
        for j in 1:i
            s = A[i, j] - sum(L[i, k] * L[j, k] for k in 1:(j - 1); init = 0.0)
            if i == j
                s <= 0 && throw(PosDefException(i))
                L[i, i] = sqrt(s)
            else
                L[i, j] = s / L[j, j]
            end
        end
    end
    return L
end

"""
    qr_gram_schmidt(A)

Thin QR via modified Gram-Schmidt: returns `(Q, R)` with orthonormal columns
in `Q` and upper-triangular `R`, so `A ≈ Q * R`.
"""
function qr_gram_schmidt(A::AbstractMatrix)
    m, n = size(A)
    Q = zeros(Float64, m, n)
    R = zeros(Float64, n, n)
    for j in 1:n
        v = float.(A[:, j])
        for i in 1:(j - 1)
            R[i, j] = dot_product(Q[:, i], v)
            v .-= R[i, j] .* Q[:, i]
        end
        R[j, j] = norm_p(v)
        R[j, j] == 0 && throw(SingularException(j))
        Q[:, j] = v ./ R[j, j]
    end
    return Q, R
end
