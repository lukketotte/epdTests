include("ep.jl")
include("aep.jl")
include("NormTest.jl")
using .NormTest, .EPmethods, .AEPmethods
using SpecialFunctions, Statistics, LinearAlgebra, Distributions, Optim
using KernelDensity, Plots, PlotThemes
import Base.rand
theme(:juno)

## RV generation not working for the AEPD
x = range(-6, 6, length = 500)
p, α = 2, 0.5

k = kde(rand(Aepd(0, 1, p, α), 2000));
k2 = kde(rand(Epd(0, 1, p), 2000));
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

## AEP normTest
x = rand(Aepd(0, 2.5, 2, 0.5), 10);

S(x, 0.0, 1.0, 0.5)
S(x, 0.0, 1.0)


R(x[3], 0., 1., .5)[1]
L(x[3], 0., 1.)[1]

length(x)

## AEP check
x = range(-6, 6, length = 500)
plot(x, pdf.(Aepd(0, 1, 5, 0.85), x))
plot!(x, pdf.(Normal(0, 1), x))

p, α = 1, 0.3
k = kde(rand(Aepd(0, 1, p, α), 1000));
x = range(-5, 5, length = 500);
plot(x, pdf(k, x))
plot!(x, pdf.(Aepd(0, 1 * 1/(2 *p^(1/p) * gamma(1 + 1/p)), p, α), x))

## MLE AEP
# α
function αL(x, p = 2, μ = 0, σ = 1)
    if x < μ
        (2 *p^(1/p) * gamma(1 + 1/p)) *(gamma(1 + 1/p) * μ-x) / (σ)
    else
        0
    end
end

function αR(x, p = 2, μ = 0, σ = 1)
    if x >= μ
        (2 *p^(1/p) * gamma(1 + 1/p)) * (gamma(1 + 1/p) * μ-x) / σ
    else
        0
    end
end

x = rand(Aepd(0, 1, 2, 0.5), 100)

La, Ra = αL.(x), αR.(x)
a = La ./ (La .+ Ra) |> sum
b = La .^2 ./ (La .+ Ra) |> sum
(a - √((a/2)^2 - b))


La .+ (La.^2 - 4*La.*(Ra .+ La)).^(0.5)


x = rand(Aepd(0, 1, p, α), 10)

αL.(x)

## Test LLN for score functions
p, nSim, n = 2, 5000, 10000
sim = 0
for i in 1:nSim
    y = rand(Epd(0, 1, p), n)
    sp, ss = S(y, mean(y), √(var(y) * (π/2)))
    β = √(var(y) * (π/2))/4
    global sim += (sum(sp) + β * sum(ss)) / n
end
sim
sim |> print


y = rand(Epd(0,1,2), 10)
S(y, mean(y), √(var(y) * (π/2)))

y = rand(Normal(0, 1), 1000)
log((abs.(y .- mean(y))) |> mean)

√(sum((y .- mean(y)).^2) / length(y))
√2 * gamma(3/2)

##
p, nSim, n = 2, 1000, 5000
# d = Epd(0, 1, p)
sims = [0. for x in 1:nSim]
for i in 1:nSim
    y = rand(Epd(0, 1, p), n)
    sims[i] = test(y, mean(y), √(var(y) * π/2))
end

y = rand(Epd(0, 1, p), 200)
√(var(y) * 2^0.5 * gamma(3/2))
√(var(y) * π/2)

k = kde(sims);
x = range(-5, 5, length = 500);
plot(x, pdf(k, x), label = "Test")
plot!(x, pdf.(Epd(0., 1., 2.), x), label = "N(0,1)")

mean(sims)


## theoretical expectation of X^2 ln|X|
using QuadGK

function f(x)
    x^2 * log(abs(x)) / √(2*π) * exp(-x^2/2)
end

a,_ = quadgk(f, -Inf, -.0001, rtol = 1e-3)
b,_ = quadgk(f, 0.0001, Inf, rtol = 1e-3)

a + b
