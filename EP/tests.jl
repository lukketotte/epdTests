include("ep.jl")
include("aep.jl")
include("NormTest.jl")
using .NormTest, .EPmethods, .AEPmethods, SpecialFunctions, Statistics, LinearAlgebra, Distributions
using KernelDensity, Plots, PlotThemes
theme(:juno)

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
