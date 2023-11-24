### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ e70ff28c-8555-11ee-3655-8b281a850fb3
using ReachabilityAnalysis, Plots

# ╔═╡ bf31bfe5-041f-4f8d-a9d6-906754805463
md"""## Introduction

We consider as an illustrative example a simple harmonic oscillator system.
This is a single degree of freedom, second order problem given by the following equation:

```math
 u''(t) + ω^2 u(t) = 0,
```
where $ω$ is a scalar and $u(t)$ is the unknown.

This problem can be associated with a spring-mass system, where $u(t)$ is the
elongation of the spring at time $t$ and the mass and stiffness set a natural
frequency $ω$. In this case we consider $ω = 4π$.

Let us introduce the state variable $v(t) := u'(t)$ and define the vector
$x(t) = [u(t), v(t)]^T$. Then we can rewrite the system in the following first order form

```math
 x'(t) = \begin{pmatrix} 0 & 1 \\ -\omega^2 & 0 \end{pmatrix} x(t)
```
"""

# ╔═╡ 1a547eec-1829-4239-8f56-45fe8d2a1f98
md"""## Reachability computation"""

# ╔═╡ 5a5d78e6-76c7-4b6c-933a-a922effb11c2
begin

# Source: https://github.com/JuliaReach/SetPropagation-FEM-Examples/tree/main/examples/SDOF

# ### Equations of motion

# Struct that holds a problem describing an harmonic oscillator with frequency ω:
# x''(t) + ω^2 x(t) = 0
#
# solution x(t) = Acos(ωt + B), v(t) = -ωAsin(ωt + B)
# x(0) = Acos(B)
# v(0) = -ωAsin(B)
#
# special case: if x(0) = A, v(0) = 0 => x(t) = A cos ωt
struct SDOF{T, ST}
    ivp::T
    X0::ST
    ω::Float64
    Ampl::Float64
    T::Float64
end

amplitude(p::SDOF) = p.Ampl
frequency(p::SDOF) = p.ω
period(p::SDOF) = p.T
MathematicalSystems.initial_state(p::SDOF) = p.X0
MathematicalSystems.InitialValueProblem(p::SDOF) = p.ivp
MathematicalSystems.state_matrix(p::SDOF) = state_matrix(p.ivp)

function analytic_solution(p::SDOF; A=p.Ampl, B=0.0, x0=nothing, v0=nothing)
    @assert p.X0 isa Singleton "the analytic solution requires a singleton initial condition"

    ω = p.ω
    if !isnothing(x0) && !isnothing(v0)
        A = sqrt(x0^2 + v0^2 / ω^2)
        B = atan(-v0 / (ω*x0))
    end
    return t -> A * cos(ω * t + B)
end

function analytic_derivative(p::SDOF; A=p.Ampl, B=0.0, x0=nothing, v0=nothing)
    @assert p.X0 isa Singleton "the analytic solution requires a singleton initial condition"

    ω = p.ω
    if !isnothing(x0) && !isnothing(v0)
        A = sqrt(x0^2 + v0^2 / ω^2)
        B = atan(-v0 / (ω*x0))
    end
     return t -> -ω * A * sin(ω * t + B)
end

function InitialValueProblem_quad(p::SDOF)
    M = hcat(1.0)
    ω = p.ω
    K = hcat(ω^2)
    C = zeros(1, 1)
    R = zeros(1)
    S = SecondOrderAffineContinuousSystem(M, C, K, R)
    @assert p.X0 isa Singleton
    x0 = element(p.X0)
    U0 = [x0[1]]
    U0dot = [x0[2]]
    return IVP(S, (U0, U0dot))
end

function sdof(; T=0.5,     # period
                Ampl=1.0,  # amplitude
                X0 = Singleton([Ampl, 0.0])) # initial condition
    ## frequency
    ω = 2π / T

    ## cast as a first-order system:
    ## x' = v
    ## v' = -ω^2 * x
    A = [ 0.0     1.0;
         -ω^2     0.0]

    prob = @ivp(X' = AX, X(0) ∈ X0)
    return SDOF(prob, X0, ω, Ampl, T)
end
end

# ╔═╡ e0d29d49-b78e-48fc-ad9b-8f9754cfdf98
@bind α html"<input type=range min=0.001 max=0.1 step=0.01>"

# ╔═╡ bbd4fee9-cdcc-4640-b69e-c2a7b2e07ee2
begin
problem = sdof()
T = problem.T

model = StepIntersect(setops=:concrete)

# alg = VREP(δ=α*T, approx_model=model)
# solvrep = solve(prob, tspan=(0.0, tmax), alg=alg);

alg = GLGM06(δ=α*T, approx_model=model)
solglg = solve(problem.ivp, tspan=(0.0, T), alg=alg);

# alg = BOX(δ=α*T, approx_model=model)
# solbox = solve(prob, tspan=(0.0, tmax), alg=alg);

# For plots: https://docs.juliaplots.org/stable/gallery/gr/generated/gr-ref016/#gr_ref016
end

# ╔═╡ 6057597d-de46-475c-8614-096ac1ff02df
md"Step size αT = $(α * T) :"

# ╔═╡ 0d99a847-c6af-4019-b955-ffa2f32492c8
plot(solglg, vars=(0, 1), xlab="t", ylab="u(t)")

# ╔═╡ 43d0a3f1-170d-4d58-9b6b-3201109d9677
plot(solglg, vars=(0, 2), xlab="t", ylab="u'(t)")

# ╔═╡ 1bd941f2-e1fa-446e-a489-4b1c87be97ed
plot(solglg, vars=(1, 2), xlab="u(t)", ylab="u'(t)")

# ╔═╡ Cell order:
# ╠═e70ff28c-8555-11ee-3655-8b281a850fb3
# ╟─bf31bfe5-041f-4f8d-a9d6-906754805463
# ╟─1a547eec-1829-4239-8f56-45fe8d2a1f98
# ╟─5a5d78e6-76c7-4b6c-933a-a922effb11c2
# ╟─6057597d-de46-475c-8614-096ac1ff02df
# ╟─e0d29d49-b78e-48fc-ad9b-8f9754cfdf98
# ╟─bbd4fee9-cdcc-4640-b69e-c2a7b2e07ee2
# ╠═0d99a847-c6af-4019-b955-ffa2f32492c8
# ╠═43d0a3f1-170d-4d58-9b6b-3201109d9677
# ╠═1bd941f2-e1fa-446e-a489-4b1c87be97ed
