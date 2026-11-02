# Descriptive statistics and ordinary least-squares regression.

"""
    mean_of(x)

Arithmetic mean of `x`.
"""
mean_of(x::AbstractVector) = sum(x) / length(x)

"""
    variance_of(x; corrected=true)

Variance of `x`; `corrected=true` divides by `n - 1` (sample variance).
"""
function variance_of(x::AbstractVector; corrected::Bool = true)
    n = length(x)
    n - corrected < 1 && throw(ArgumentError("need at least 2 values for sample variance"))
    μ = mean_of(x)
    return sum((v - μ)^2 for v in x) / (n - (corrected ? 1 : 0))
end

"""Standard deviation of `x` (see [`variance_of`](@ref))."""
std_of(x::AbstractVector; corrected::Bool = true) = sqrt(variance_of(x; corrected = corrected))

"""
    median_of(x)

Median of `x` (average of the two middle values for even length).
"""
function median_of(x::AbstractVector)
    s = sort(collect(x))
    n = length(s)
    n == 0 && throw(ArgumentError("median of an empty vector is undefined"))
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

"""
    quantile_of(x, q)

`q`-quantile of `x` (`0 ≤ q ≤ 1`) by linear interpolation between
closest ranks.
"""
function quantile_of(x::AbstractVector, q::Real)
    0 <= q <= 1 || throw(ArgumentError("q must be in [0, 1]"))
    s = sort(collect(x))
    n = length(s)
    n == 0 && throw(ArgumentError("quantile of an empty vector is undefined"))
    pos = 1 + q * (n - 1)
    lo = floor(Int, pos)
    hi = ceil(Int, pos)
    lo == hi && return s[lo]
    return s[lo] + (pos - lo) * (s[hi] - s[lo])
end

"""
    covariance_of(x, y; corrected=true)

Covariance of two equal-length vectors.
"""
function covariance_of(x::AbstractVector, y::AbstractVector; corrected::Bool = true)
    length(x) == length(y) || throw(DimensionMismatch("vectors must have equal length"))
    n = length(x)
    n - corrected < 1 && throw(ArgumentError("need at least 2 values"))
    return sum((x[i] - mean_of(x)) * (y[i] - mean_of(y)) for i in 1:n) / (n - (corrected ? 1 : 0))
end

"""
    correlation_of(x, y)

Pearson correlation coefficient in `[-1, 1]`.
"""
function correlation_of(x::AbstractVector, y::AbstractVector)
    sx, sy = std_of(x), std_of(y)
    (sx == 0 || sy == 0) && throw(ArgumentError("correlation is undefined for constant data"))
    return covariance_of(x, y) / (sx * sy)
end

"""
    zscores(x)

Standard scores `(x[i] - mean) / std` of `x`.
"""
function zscores(x::AbstractVector)
    μ, σ = mean_of(x), std_of(x)
    σ == 0 && throw(ArgumentError("z-scores are undefined for constant data"))
    return [(v - μ) / σ for v in x]
end

"""
    RegressionResult

Result of [`linear_regression`](@ref): `slope`, `intercept`, and `r_squared`.
"""
struct RegressionResult
    slope::Float64
    intercept::Float64
    r_squared::Float64
end

"""
    linear_regression(x, y)

Ordinary least-squares fit of `y ≈ slope * x + intercept`, with the
coefficient of determination `r_squared`.
"""
function linear_regression(x::AbstractVector, y::AbstractVector)
    length(x) == length(y) || throw(DimensionMismatch("vectors must have equal length"))
    length(x) >= 2 || throw(ArgumentError("need at least 2 points"))
    mx, my = mean_of(x), mean_of(y)
    sxx = sum((v - mx)^2 for v in x)
    sxx == 0 && throw(ArgumentError("regression is undefined for constant x"))
    slope = sum((x[i] - mx) * (y[i] - my) for i in 1:length(x)) / sxx
    intercept = my - slope * mx
    ss_res = sum((y[i] - (slope * x[i] + intercept))^2 for i in 1:length(x))
    ss_tot = sum((v - my)^2 for v in y)
    r2 = ss_tot == 0 ? 1.0 : 1 - ss_res / ss_tot
    return RegressionResult(slope, intercept, r2)
end
