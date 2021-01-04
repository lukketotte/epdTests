include("ep.jl")
include("aep.jl")
include("NormTest.jl")
using .NormTest, .EPmethods, .AEPmethods
using SpecialFunctions, Statistics, LinearAlgebra, Distributions, Optim
using KernelDensity, Plots, PlotThemes
import Base.rand
theme(:juno)


## EPD mle
function loglikEPD(θ, p, x) where {T <: Real}
    μ, σ = θ
    σ = exp(σ)
    -log.(pdf.(Epd(μ, σ, p), x)) |> sum
end

p = 2.
x = rand(Epd(0, 1, p), 1000)

func = TwiceDifferentiable(vars -> loglikEPD(vars, p, x), ones(2), autodiff =:forward)
optimum = optimize(func, [0., log(0.7)])
Optim.converged(optimum) || throw(ConvergenceError("Optimizer did not converge"))
mle = Optim.minimizer(optimum)
mle[2] = exp(mle[2])

mle[2]
√var(x)
function loglik(θ, p, x)
    μ, σ, α = θ
    σ = exp(σ)
    -log.(pdf.(Aepd(μ, σ, p, α), x)) |> sum
end

func = TwiceDifferentiable(vars -> loglik(vars, p, x), ones(3), autodiff =:forward)
optimum = optimize(func, [0., log(0.7), 0.5])
Optim.converged(optimum) || throw(ConvergenceError("Optimizer did not converge"))
mle = Optim.minimizer(optimum)

## RV generation not working for the AEPD
x = range(-6, 6, length = 500)
p, α = 2, 0.5

k = kde(rand(Aepd(0, 2, p, α), 2000));
k2 = kde(rand(Epd(0, 2, p), 2000));
x = range(-5, 5, length = 500);
plot(x, pdf(k, x), label = "aepd")
plot!(x, pdf(k2, x), label = "epd")


## AEP MLE
function loglike(θ, p, x)
    μ, σ, α = θ
    σ = exp(σ)
    -log.(pdf.(Aepd(μ, σ, p, α), x)) |> sum
end

x = rand(Aepd(0, 1, 2, 0.5), 1000);
params0 = [0.1, log(1.1), 0.5];
mle = MLE(params0, 2., x)
p = 2
mle[2] / (1/(2 * p^(1/p) * gamma(1 + 1/p)))

((p^(1/p) * gamma(1 + 1/p)))
π/2

optimum = optimize(b -> loglike(b, 2, x), params0, BFGS())
mle = Optim.minimizer(optimum)
mle[2] = exp(mle[2])
mle
Optim.converged(optimum)
@warn("not working")

Optim.minimizer(optimum)
typeof(params0) <: Array{T} where {T <: Real}

p = 2
mle[2] / (1/(2 * p^(1/p) * gamma(1 + 1/p)))

typeof(params0) <: Array{Float64, 1}
