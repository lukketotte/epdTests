include("..\\EP\\aep.jl")
using .AEPmethods, SpecialFunctions, Statistics, LinearAlgebra, Distributions, Roots
using Plots, PlotThemes
theme(:juno)

# https://github.com/JuliaMath/Roots.jl

function f(α, x = -0.55, μ = 0., σ = 1., p = 2.)
    L = x < μ ? gamma(1+1/p)*(μ - x) / σ : 0
    R = x >= μ ? gamma(1+1/p)*(x - μ) / σ : 0
    p/α^2 * L^p - p/(1-α)^2 * R^p
end


α = range(0, 1, length = 300)
plot(α, f.(α))


f(x) = x^5 -x  +0.5

find_zero(f, (0,1))

a = true ? 1 : 0
