module AEPmethods

export Aepd, MLE

using Distributions, SpecialFunctions, Random, Optim, Logging
import Distributions.pdf, Base.rand

include("structs.jl")
using .Structs


struct Aepd <: ContinuousUnivariateDistribution
    mu::Real
    sigma::Real
    p::Real
    alpha::Real
    Aepd(mu, sigma, p, alpha) = new(Real(mu), Real(sigma), Real(p), Real(alpha))
end

function pdf(d::Aepd, x::Real)
    μ, σ, p, α = d.mu, d.sigma, d.p, d.alpha
    K = 1/(2 * p^(1/p) * gamma(1 + 1/p))
    if x <= μ
        K * exp(-abs((x-μ)/(2*α*σ))^p / p) / σ
    else
        K * exp(-abs((x-μ)/(2*(1-α)*σ))^p / p) / σ
    end
end

function rand(rng::AbstractRNG, d::Aepd)
    μ, σ, p, α = d.mu, d.sigma, d.p, d.alpha
    # rescale it back
    σ = σ * 2*p^(1/p)*gamma(1 + 1/p)
    u = rand(rng)
    W = rand(Gamma(1/p, 1))
    S = sign(u - α)
    μ + α*σ*(S - 1) / (2*gamma(1 + 1/p)) * W^(1/p) + (1-α)*σ*(S + 1) / (2*gamma(1 + 1/p)) * W^(1/p)
end

function loglik(θ, p, x)
    μ, σ, α = θ
    σ = exp(σ)
    -log.(pdf.(Aepd(μ, σ, p, α), x)) |> sum
end

function MLE(θ::Array{T, 1}, p::T, x::Array{T, 1}) where {T <: Real}
    length(θ) === 3 || throw(ArgumentError("θ not of length 3"))
    func = TwiceDifferentiable(vars -> loglik(vars, 2., x), ones(3), autodiff =:forward)
    optimum = optimize(func, θ)
    Optim.converged(optimum) || throw(ConvergenceError("Optimizer did not converge"))
    mle = Optim.minimizer(optimum)
    mle[2] = exp(mle[2])
    mle
end


end
