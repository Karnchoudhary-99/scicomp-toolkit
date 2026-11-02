# Vector utilities, implemented from scratch (no external packages).

"""
    dot_product(a, b)

Dot product of two equal-length vectors. Throws `DimensionMismatch` on
length mismatch.
"""
function dot_product(a::AbstractVector, b::AbstractVector)
    length(a) == length(b) || throw(DimensionMismatch("vectors must have equal length"))
    s = zero(promote_type(eltype(a), eltype(b)))
    for i in eachindex(a, b)
        s += a[i] * b[i]
    end
    return s
end

"""
    norm_p(a, p=2)

The ``p``-norm of `a` (`p = 2` Euclidean, `p = 1` Manhattan, `p = Inf`
max-absolute).
"""
function norm_p(a::AbstractVector, p::Real = 2)
    if p == 2
        return sqrt(sum(abs2, a))
    elseif p == 1
        return sum(abs, a)
    elseif p == Inf
        return isempty(a) ? zero(eltype(a)) : maximum(abs, a)
    else
        return sum(abs.(a) .^ p)^(one(p) / p)
    end
end

"""
    cross_product(a, b)

Cross product of two 3-vectors.
"""
function cross_product(a::AbstractVector, b::AbstractVector)
    length(a) == 3 && length(b) == 3 ||
        throw(DimensionMismatch("cross product is defined for 3-vectors"))
    return [
        a[2] * b[3] - a[3] * b[2],
        a[3] * b[1] - a[1] * b[3],
        a[1] * b[2] - a[2] * b[1],
    ]
end

"""
    angle_between(a, b)

Angle in radians between two non-zero vectors, clamped to `[0, π]`.
"""
function angle_between(a::AbstractVector, b::AbstractVector)
    na, nb = norm_p(a), norm_p(b)
    (na == 0 || nb == 0) && throw(ArgumentError("angle is undefined for zero vectors"))
    return acos(clamp(dot_product(a, b) / (na * nb), -1, 1))
end

"""
    project_onto(a, b)

Orthogonal projection of `a` onto the non-zero vector `b`.
"""
function project_onto(a::AbstractVector, b::AbstractVector)
    denom = dot_product(b, b)
    denom == 0 && throw(ArgumentError("cannot project onto the zero vector"))
    return (dot_product(a, b) / denom) .* b
end

"""
    normalize_vector(a)

Unit vector in the direction of `a`.
"""
function normalize_vector(a::AbstractVector)
    n = norm_p(a)
    n == 0 && throw(ArgumentError("cannot normalize the zero vector"))
    return a ./ n
end

"""
    outer_product(a, b)

Matrix `M` with `M[i, j] == a[i] * b[j]`.
"""
function outer_product(a::AbstractVector, b::AbstractVector)
    return [a[i] * b[j] for i in eachindex(a), j in eachindex(b)]
end
