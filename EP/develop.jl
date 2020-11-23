# Current: develop of AEP methods for norm test
include("ep.jl")
include("aep.jl")
include("NormTest.jl")
using .NormTest, .EPmethods, .AEPmethods
using SpecialFunctions, Statistics, LinearAlgebra, Distributions, Optim

x = rand(Aepd(0, 2.5, 2, 0.5), 10);
p, σ, μ, α = 2., 1., 0., 0.5

test(x, μ, σ)
test(x, μ, σ, α)

## size
x = rand(Aepd(0, 2.5, 2, 0.7), 100);
mean(x)

var(x) * π/2

u, s, a = MLEs([mean(x), var(x), 0.8], 2., x);
s * π/2


function loglik(θ, p, x)
    μ, σ, α = θ
    σ = exp(σ)
    -log.(pdf.(Aepd(μ, σ, p, α), x)) |> sum
end

x = rand(Aepd(0, 1, 2, 0.8), 5000);
func = TwiceDifferentiable(vars -> loglik(vars, 2., x), ones(3), autodiff =:forward)

opt = optimize(func, [mean(x), log(var(x)), 0.7])
mles = Optim.minimizer(opt)
mles[2] = exp(mles[2])
println(mles)

function MLEs(θ::Array{T, 1}, p::T, x::Array{T, 1}) where {T <: Real}
    length(θ) === 3 || throw(ArgumentError("θ not of length 3"))
    optimum = optimize(b -> loglik(b, p, x), θ)
    Optim.converged(optimum) || @warn("Not converged")
    mle = Optim.minimizer(optimum)
    mle[2] = exp(mle[2])
    mle
end
