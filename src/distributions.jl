# Common probability distributions (dependency-free, stdlib only).
#
# The normal CDF uses the Abramowitz–Stegun 7.1.26 rational approximation
# (|error| < 1.5e-7), since Julia's Base does not ship `erf` and this
# package takes no external dependencies.

"""
    normal_pdf(x; mu=0, sigma=1)

Density of the normal distribution at `x`.
"""
function normal_pdf(x::Real; mu::Real = 0, sigma::Real = 1)
    sigma > 0 || throw(ArgumentError("sigma must be positive"))
    z = (x - mu) / sigma
    return exp(-z^2 / 2) / (sigma * sqrt(2π))
end

"""
    normal_cdf(x; mu=0, sigma=1)

CDF of the normal distribution at `x` (Abramowitz–Stegun approximation).
"""
function normal_cdf(x::Real; mu::Real = 0, sigma::Real = 1)
    sigma > 0 || throw(ArgumentError("sigma must be positive"))
    z = (x - mu) / sigma
    t = 1 / (1 + 0.2316419 * abs(z))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    p = 1 - exp(-z^2 / 2) / sqrt(2π) * poly
    return z >= 0 ? p : 1 - p
end

"""
    uniform_pdf(x, a, b) / uniform_cdf(x, a, b)

Density / CDF of the continuous uniform distribution on `[a, b]`.
"""
function uniform_pdf(x::Real, a::Real, b::Real)
    a < b || throw(ArgumentError("need a < b"))
    return a <= x <= b ? 1 / (b - a) : 0.0
end

function uniform_cdf(x::Real, a::Real, b::Real)
    a < b || throw(ArgumentError("need a < b"))
    return x < a ? 0.0 : x > b ? 1.0 : (x - a) / (b - a)
end

"""
    exponential_pdf(x, lambda) / exponential_cdf(x, lambda)

Density / CDF of the exponential distribution with rate `lambda`.
"""
function exponential_pdf(x::Real, lambda::Real)
    lambda > 0 || throw(ArgumentError("lambda must be positive"))
    return x < 0 ? 0.0 : lambda * exp(-lambda * x)
end

function exponential_cdf(x::Real, lambda::Real)
    lambda > 0 || throw(ArgumentError("lambda must be positive"))
    return x < 0 ? 0.0 : 1 - exp(-lambda * x)
end

_logfactorial(n::Integer) = sum(log, 1:n; init = 0.0)

"""
    binomial_pmf(k, n, p)

Probability of exactly `k` successes in `n` Bernoulli trials with
success probability `p`.
"""
function binomial_pmf(k::Integer, n::Integer, p::Real)
    0 <= p <= 1 || throw(ArgumentError("p must be in [0, 1]"))
    0 <= k <= n || return 0.0
    logpmf = _logfactorial(n) - _logfactorial(k) - _logfactorial(n - k) +
             k * log(p) + (n - k) * log(1 - p)
    return exp(logpmf)
end

"""
    poisson_pmf(k, lambda)

Probability of exactly `k` events for a Poisson rate `lambda`.
"""
function poisson_pmf(k::Integer, lambda::Real)
    lambda > 0 || throw(ArgumentError("lambda must be positive"))
    k < 0 && return 0.0
    return exp(k * log(lambda) - lambda - _logfactorial(k))
end
