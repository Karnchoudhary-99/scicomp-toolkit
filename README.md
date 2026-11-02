# Scientific Computing Toolkit

Small, dependency-free numerical routines in Julia — linear algebra,
statistics, distributions, and numerical methods, with tests and examples.
Only Julia's standard library is used (LinearAlgebra, Test); no external
packages are required.

## Quick start

```julia
using Pkg
Pkg.add(url = "https://github.com/Karnchoudhary-99/scicomp-toolkit")
using SciCompToolkit

gauss_solve([2.0 1.0; 5.0 7.0], [11.0, 13.0])  # [7.111, -3.222]
normal_cdf(1.96)                                # 0.975
simpson(x -> x^2, 0, 1, 100)                    # 0.3333
rk4_solve((t, y) -> -y, 1.0, 0.0, 1.0, 0.01)[2][end]  # ≈ 1/e
```

Run the demo tour and the test suite:

```sh
julia --project=. examples/demo.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

## What's inside

| Area | Routines |
| --- | --- |
| Vectors & matrices | dot/norm/cross/angle/projection, outer product, trace, Frobenius norm, Hilbert & Vandermonde test matrices |
| Decompositions | LU with partial pivoting, Cholesky, thin QR (modified Gram–Schmidt) |
| Solvers | Gaussian elimination, Cholesky solve, Jacobi, Gauss–Seidel |
| Eigenvalues | Power iteration, inverse iteration, Rayleigh quotient |
| Statistics | mean/variance/std/median/quantiles, covariance, Pearson correlation, z-scores, least-squares regression with R² |
| Distributions | Normal (pdf/cdf), uniform, exponential, binomial, Poisson |
| Calculus | Central differences, gradient, trapezoid & Simpson integration |
| Roots | Bisection, Newton, secant |
| ODEs | Explicit Euler, classic RK4 (scalar or vector state) |
| Optimization | Gradient descent, Newton minimization |

Notes on accuracy: the normal CDF uses the Abramowitz–Stegun 7.1.26
approximation (|error| < 1.5e-7) because Base Julia does not ship `erf`
and this package takes no external dependencies.

## Roadmap

- [x] Project scaffold (README roadmap, license, gitignore)
- [x] Vector and matrix operations
- [x] Linear algebra (solvers, decompositions)
- [x] Statistics and distributions
- [x] Numerical methods (integration, root finding, ODEs)
- [ ] Self-review checklist, then public

## License

MIT — see [LICENSE](LICENSE).
