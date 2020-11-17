include("EP\\NormTest.jl")

using Distributions, Plots, PlotThemes, KernelDensity, SpecialFunctions, .NormTest
theme(:juno)

function A(y::Array{T, 1}, σ::T, μ::T) where {T <: Real}
    n = length(y)
    √n * mean(((y .- μ)/σ).^2)
end

n, nsims = 1000, 20000
sims = [0. for x in 1:nsims]
for i in 1:nsims
    y = rand(Normal(0, 1), n)
    sims[i] = (digamma(3/2) + 1) / 0.5 * gamma(3/2)^2 * √n * mean((y .- mean(y)).^2) / √var(y)
end

k = kde(sims);
x = range(minimum(sims), maximum(sims), length = 500);
plot(x, pdf(k, x))
mean(sims)

gamma(3/2)^2
digamma(3/2) / 0.5

p = range(0.1, 100, length = 500);

function f(p)
    1/p^3 * (1 + 1/p) * digamma(1 + 1/p)
end

function g(p)
    gamma(1/p) * gamma(2 - 1/p)
end

plot(p, g.(p))

## determinant of information matrix
using LinearAlgebra
IM(1, .8, 1., 0.) |> det
det(I)

## Lipschitz constant of the log likelihood
function Lₚ(y::T, μ::T, p::T, σ::T) where {T <: Real}
    return abs(y-μ) * gamma(1 + 1/p) / σ
end
function Scores(y::T, μ::T, p::T, σ::T) where {T <: Real}
    ℓ = Lₚ(y, μ, p, σ)

    Sₚ = ℓ^p * (1/p * digamma(1 + 1/p) - log(ℓ))
    Sᵤ = (gamma(1/p) / σ) * ℓ^(p-1) * sign(μ-y)
    Sₛ = p/σ * ℓ^p - 1/σ

    return Sₚ, Sᵤ, Sₛ
end

function InnerScore(y::T, μ::T, p::T, σ::T) where {T <: Real}
    √(Scores(y, μ, p, σ).^2 |> sum)
end

function h(μ, σ)
    InnerScore(0., μ, 1., σ)
end

function g(p, σ)
    InnerScore(0.5, 0., p, σ)
end

surface(μ,σ,h,camera=(60,30), size=[800,500])
surface(p,σ,g,camera=(60,30), size=[800,500])


p = range(0.51, 3, length = 200);
σ = range(1, 5, length = 200);
μ = range(-2, 2, length = 200);
x = similar(p)
y = similar(p)
z = similar(p)


for i in 1:200
    x[i] = InnerScore(0.5, 0., p[i], 1.)
    y[i] = InnerScore(0.5, 0., 1., σ[i])
    z[i] = InnerScore(0.5, μ[i], 1., 1.)
end

plot(p, x)
plot(σ, y)
plot(μ, z)

##
n = 100
ts = range(0, stop = 8π, length = n)
x = ts .* map(cos, ts)
y = (0.1ts) .* map(sin, ts)
z = 1:n
plot(x, y, z, zcolor = reverse(z), m = (10, 0.8, :blues, Plots.stroke(0)), leg = false, cbar = true, w = 5)
plot!(zeros(n), zeros(n), 1:n, w = 10)

surface(x, y, z)

InnerScore(0.5, 0., 0.6, 1.)
InnerScore(0.5, 20., 0.6, 10.)


function h(σ, p)
    Lₚ(0., 0.5, p, σ)
end

function h(σ, μ)
    Lₚ(0., μ, 0.55, σ) |> exp
end

surface(σ, μ, h)
