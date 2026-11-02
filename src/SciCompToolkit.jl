# SciCompToolkit: small, dependency-free numerical routines in Julia.
module SciCompToolkit

using LinearAlgebra: I, SingularException, PosDefException

include("vectors.jl")
include("matrices.jl")
include("decompositions.jl")
include("solvers.jl")
include("eigen.jl")
include("stats.jl")
include("distributions.jl")
include("calculus.jl")
include("roots.jl")
include("ode.jl")
include("optimize.jl")

export dot_product, norm_p, cross_product, angle_between, project_onto,
    normalize_vector, outer_product
export frobenius_norm, matrix_trace, is_symmetric, diag_vector, diag_matrix,
    hilbert_matrix, vandermonde
export LUFactor, lu_factor, cholesky_factor, qr_gram_schmidt
export forward_substitution, back_substitution, gauss_solve, cholesky_solve,
    jacobi_solve, gauss_seidel_solve
export rayleigh_quotient, power_iteration, inverse_iteration
export mean_of, variance_of, std_of, median_of, quantile_of, covariance_of,
    correlation_of, zscores, RegressionResult, linear_regression
export normal_pdf, normal_cdf, uniform_pdf, uniform_cdf, exponential_pdf,
    exponential_cdf, binomial_pmf, poisson_pmf
export central_difference, second_difference, gradient_central, trapezoid, simpson
export RootResult, bisection, newton_root, secant_root
export euler_solve, rk4_step, rk4_solve
export OptimizeResult, gradient_descent, newton_minimize

end # module SciCompToolkit
