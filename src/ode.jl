# Initial-value ODE solvers: explicit Euler and classic RK4.
#
# `f(t, y)` returns the derivative at time `t` and state `y` (a scalar or
# vector). Solvers return `(ts, ys)` where `ys[i]` is the state at `ts[i]`.

function _ode_steps(t0::Real, t1::Real, dt::Real)
    dt > 0 || throw(ArgumentError("dt must be positive"))
    t0 < t1 || throw(ArgumentError("need t0 < t1"))
    return max(1, round(Int, (t1 - t0) / dt))
end

"""
    euler_solve(f, y0, t0, t1, dt)

Explicit Euler integration of `y' = f(t, y)` from `t0` to `t1`.
"""
function euler_solve(f::Function, y0, t0::Real, t1::Real, dt::Real)
    n = _ode_steps(t0, t1, dt)
    h = (t1 - t0) / n
    ts = [t0 + i * h for i in 0:n]
    ys = Vector{typeof(y0)}(undef, n + 1)
    ys[1] = y0
    for i in 1:n
        ys[i + 1] = ys[i] .+ h .* f(ts[i], ys[i])
    end
    return ts, ys
end

"""
    rk4_step(f, t, y, h)

One classic 4th-order Runge–Kutta step.
"""
function rk4_step(f::Function, t::Real, y, h::Real)
    k1 = f(t, y)
    k2 = f(t + h / 2, y .+ (h / 2) .* k1)
    k3 = f(t + h / 2, y .+ (h / 2) .* k2)
    k4 = f(t + h, y .+ h .* k3)
    return y .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

"""
    rk4_solve(f, y0, t0, t1, dt)

Classic RK4 integration of `y' = f(t, y)` from `t0` to `t1`.
"""
function rk4_solve(f::Function, y0, t0::Real, t1::Real, dt::Real)
    n = _ode_steps(t0, t1, dt)
    h = (t1 - t0) / n
    ts = [t0 + i * h for i in 0:n]
    ys = Vector{typeof(y0)}(undef, n + 1)
    ys[1] = y0
    for i in 1:n
        ys[i + 1] = rk4_step(f, ts[i], ys[i], h)
    end
    return ts, ys
end
