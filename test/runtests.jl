using Test
using SciCompToolkit
using LinearAlgebra: I, SingularException

@testset "vectors" begin
    @test dot_product([1, 2, 3], [4, 5, 6]) == 32
    @test norm_p([3, 4]) ≈ 5
    @test norm_p([1, -2, 3], 1) == 6
    @test norm_p([1, -5, 3], Inf) == 5
    @test cross_product([1, 0, 0], [0, 1, 0]) == [0, 0, 1]
    @test angle_between([1, 0], [0, 1]) ≈ π / 2
    @test project_onto([3, 4], [1, 0]) ≈ [3, 0]
    @test normalize_vector([3, 4]) ≈ [0.6, 0.8]
    @test outer_product([1, 2], [3, 4]) == [3 4; 6 8]
    @test_throws DimensionMismatch dot_product([1], [1, 2])
    @test_throws ArgumentError normalize_vector([0, 0])
end

@testset "matrices" begin
    @test frobenius_norm([3 4; 0 0]) ≈ 5
    @test matrix_trace([1 2; 3 4]) == 5
    @test is_symmetric([1 2; 2 1])
    @test !is_symmetric([1 2; 3 4])
    @test diag_vector([1 2; 3 4]) == [1, 4]
    @test diag_matrix([2, 3]) == [2 0; 0 3]
    @test hilbert_matrix(2) ≈ [1 1 / 2; 1 / 2 1 / 3]
    @test vandermonde([1, 2, 3]) == [1 1 1; 1 2 4; 1 3 9]
end

@testset "decompositions and solvers" begin
    A = [2.0 1.0; 5.0 7.0]
    b = [11.0, 13.0]
    x = gauss_solve(A, b)
    @test A * x ≈ b
    F = lu_factor(A)
    @test F.L * F.U ≈ A[F.p, :]
    @test back_substitution(F.U, forward_substitution(F.L, b[F.p])) ≈ x
    S = [4.0 2.0; 2.0 3.0]
    L = cholesky_factor(S)
    @test L * L' ≈ S
    @test cholesky_solve(S, [6.0, 5.0]) ≈ S \ [6.0, 5.0]
    Q, R = qr_gram_schmidt([1.0 1.0; 1.0 -1.0; 1.0 2.0])
    @test Q' * Q ≈ I
    @test Q * R ≈ [1.0 1.0; 1.0 -1.0; 1.0 2.0]
    D = [4.0 1.0; 1.0 3.0]
    xj, _, cj = jacobi_solve(D, [1.0, 2.0])
    @test cj && D * xj ≈ [1.0, 2.0]
    xg, _, cg = gauss_seidel_solve(D, [1.0, 2.0])
    @test cg && D * xg ≈ [1.0, 2.0]
    @test_throws SingularException gauss_solve([1.0 2.0; 2.0 4.0], [1.0, 1.0])
end

@testset "eigenvalues" begin
    A = [2.0 0.0; 0.0 1.0]
    λ, v, _, ok = power_iteration(A)
    @test ok && λ ≈ 2 && abs(v[1]) ≈ 1
    λ2, _, _, ok2 = inverse_iteration([2.0 0.0; 0.0 5.0], 4.6)
    @test ok2 && λ2 ≈ 5
    @test rayleigh_quotient(A, [1.0, 0.0]) ≈ 2
end

@testset "statistics" begin
    x = [1.0, 2, 3, 4, 5]
    @test mean_of(x) == 3
    @test variance_of(x) ≈ 2.5
    @test variance_of(x; corrected = false) ≈ 2.0
    @test std_of(x) ≈ sqrt(2.5)
    @test median_of(x) == 3
    @test median_of([1, 2, 3, 4]) == 2.5
    @test quantile_of(x, 0) == 1 && quantile_of(x, 1) == 5
    @test quantile_of(x, 0.5) == 3
    @test covariance_of(x, 2 .* x) ≈ 5.0
    @test correlation_of(x, 2 .* x) ≈ 1
    @test correlation_of(x, -x) ≈ -1
    r = linear_regression([0.0, 1, 2], [1.0, 3, 5])
    @test r.slope ≈ 2 && r.intercept ≈ 1 && r.r_squared ≈ 1
    @test zscores([0.0, 2.0]) ≈ [-1 / sqrt(2), 1 / sqrt(2)]
end

@testset "distributions" begin
    @test normal_pdf(0) ≈ 1 / sqrt(2π)
    @test normal_cdf(0) ≈ 0.5 atol = 1e-6
    @test normal_cdf(1.96) ≈ 0.975 atol = 1e-3
    @test normal_cdf(-1.96) ≈ 0.025 atol = 1e-3
    @test uniform_pdf(0.5, 0, 1) == 1 && uniform_pdf(2, 0, 1) == 0
    @test uniform_cdf(0.25, 0, 1) == 0.25
    @test exponential_pdf(0, 2.0) == 2.0
    @test exponential_cdf(0, 2.0) == 0
    @test binomial_pmf(0, 1, 0.5) ≈ 0.5
    @test binomial_pmf(2, 2, 0.5) ≈ 0.25
    @test binomial_pmf(5, 3, 0.5) == 0.0
    @test poisson_pmf(0, 1.0) ≈ exp(-1)
    @test poisson_pmf(1, 2.0) ≈ 2 * exp(-2)
end

@testset "calculus" begin
    @test central_difference(x -> x^2, 3.0) ≈ 6 atol = 1e-4
    @test second_difference(x -> x^3, 2.0) ≈ 12 atol = 1e-3
    @test gradient_central(v -> v[1]^2 + 3 * v[2]^2, [1.0, 2.0]) ≈ [2, 12] atol = 1e-4
    @test trapezoid(x -> x^2, 0, 1, 1000) ≈ 1 / 3 atol = 1e-6
    @test simpson(x -> x^2, 0, 1, 100) ≈ 1 / 3 atol = 1e-9
    @test simpson(x -> x^3, 0, 2, 10) ≈ 4 atol = 1e-9
    @test_throws ArgumentError simpson(x -> x, 0, 1, 3)
end

@testset "roots" begin
    r = bisection(x -> x^2 - 2, 1.0, 2.0)
    @test r.converged
    @test r.root ≈ sqrt(2) atol = 1e-9
    r = newton_root(x -> x^2 - 2, x -> 2x, 1.0)
    @test r.converged
    @test r.root ≈ sqrt(2) atol = 1e-9
    r = secant_root(x -> x^2 - 2, 1.0, 2.0)
    @test r.converged
    @test r.root ≈ sqrt(2) atol = 1e-9
    @test_throws ArgumentError bisection(x -> x^2 + 1, -1.0, 1.0)
end

@testset "ode" begin
    ts, ys = rk4_solve((t, y) -> -y, 1.0, 0.0, 1.0, 0.01)
    @test ys[end] ≈ exp(-1) atol = 1e-6
    @test length(ts) == length(ys) == 101
    # Harmonic oscillator: y'' = -y as a first-order system.
    f = (t, y) -> [y[2], -y[1]]
    _, ys2 = rk4_solve(f, [1.0, 0.0], 0.0, 2π, 0.01)
    @test ys2[end][1] ≈ 1 atol = 1e-3
    @test ys2[end][2] ≈ 0 atol = 1e-3
    _, ys3 = euler_solve((t, y) -> -y, 1.0, 0.0, 1.0, 0.001)
    @test ys3[end] ≈ exp(-1) atol = 1e-2
end

@testset "optimization" begin
    r = gradient_descent(x -> (x - 3)^2, x -> 2 * (x - 3), 0.0; lr = 0.1)
    @test r.converged
    @test r.x ≈ 3 atol = 1e-4
    r = gradient_descent(v -> sum((v .- [1.0, 2.0]) .^ 2), v -> 2 .* (v .- [1.0, 2.0]), [0.0, 0.0]; lr = 0.1)
    @test r.converged
    @test r.x ≈ [1.0, 2.0] atol = 1e-4
    r = newton_minimize(x -> (x - 2)^2 + 1, x -> 2 * (x - 2), x -> 2.0, 0.0)
    @test r.converged
    @test r.x ≈ 2 atol = 1e-9
    @test r.fmin ≈ 1
end
