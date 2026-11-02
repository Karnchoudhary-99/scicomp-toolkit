# Demo tour of SciCompToolkit. Run with: julia --project=. examples/demo.jl
using SciCompToolkit

println("== linear algebra ==")
A = [2.0 1.0; 5.0 7.0]
println("solve A*x = [11, 13] -> ", gauss_solve(A, [11.0, 13.0]))
F = lu_factor(A)
println("LU check P*A == L*U: ", F.L * F.U ≈ A[F.p, :])
λ, v, _, _ = power_iteration([2.0 0.0; 0.0 1.0])
println("dominant eigenvalue of diag(2, 1): ", round(λ, digits = 6))

println("\n== statistics ==")
x = [1.0, 2, 3, 4, 5]
println("mean/std of 1..5: ", mean_of(x), " / ", round(std_of(x), digits = 4))
r = linear_regression([0.0, 1, 2], [1.0, 3, 5])
println("fit y = 2x + 1: slope=", r.slope, " intercept=", r.intercept, " R²=", r.r_squared)

println("\n== distributions ==")
println("P(Z <= 1.96) = ", round(normal_cdf(1.96), digits = 4), " (expect ~0.975)")

println("\n== calculus & roots ==")
println("∫₀¹ x² dx (Simpson) = ", simpson(t -> t^2, 0, 1, 100))
println("√2 by bisection = ", bisection(t -> t^2 - 2, 1.0, 2.0).root)

println("\n== ODEs & optimization ==")
_, ys = rk4_solve((t, y) -> -y, 1.0, 0.0, 1.0, 0.01)
println("y' = -y, y(1) = ", round(ys[end], digits = 6), " (expect 1/e = ", round(exp(-1), digits = 6), ")")
opt = gradient_descent(t -> (t - 3)^2, t -> 2 * (t - 3), 0.0; lr = 0.1)
println("min of (x-3)² at x = ", round(opt.x, digits = 6))
println("\nDemo complete.")
