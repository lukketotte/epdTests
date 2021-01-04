module EPmethods

export Epd, MleEpd

using Distributions, SpecialFunctions, Random, Optim
import Distributions.pdf, Distributions.quantile, Base.rand

include("structs.jl")
using .Structs

struct Epd <: ContinuousUnivariateDistribution
    mu::Real
    sigma::Real
    p::Real
    Epd(mu, sigma, p) = new(Real(mu), Real(sigma), Real(p))
end

function pdf(d::Epd, x::Real)
    μ, σ, p = d.mu, d.sigma, d.p
    K = σ * 2 * p^(1/p) * gamma(1 + 1/p)
    exp(-1/p * abs((x - μ)/σ)^p)/K
end

function quantile(d::Epd, x::Real)
    ν, μ, σ, p = 2*x-1, d.mu, d.sigma, d.p
    G = quantile(Gamma(1/p, σ^p), abs(ν))
    sign(ν) * (p*G)^(1/p)
end

function rand(rng::AbstractRNG, d::Epd)
    r = rand(rng)
    quantile(d, r) + d.mu
end


## MLE for p != 2, not sure how to do this within distributions
function loglikEPD(θ, p, x) where {T <: Real}
    μ, σ = θ
    σ = exp(σ)
    -log.(pdf.(Epd(μ, σ, p), x)) |> sum
end

function MleEpd(θ::Array{T, 1}, p::T, x::Array{T, 1}) where {T <: Real}
    length(θ) === 2 || throw(ArgumentError("θ not of length 2"))
    func = TwiceDifferentiable(vars -> loglikEPD(vars, p, x), ones(2), autodiff =:forward)
    optimum = optimize(func, θ)
    Optim.converged(optimum) || throw(ConvergenceError("Optimizer did not converge"))
    mle = Optim.minimizer(optimum)
    mle[2] = exp(mle[2])
    mle
end

end
