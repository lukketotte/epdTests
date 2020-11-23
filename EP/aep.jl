module AEPmethods

export Aepd, MLE

using Distributions, SpecialFunctions, Random, Optim, Logging
import Distributions.pdf, Base.rand

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
    # OBS: σ = σ / K_{EP}(p) here
    μ, σ, p, α = d.mu, d.sigma, d.p, d.alpha
    u = rand(rng)
    W1, W2 = rand(Gamma(1/p, 1), 2)
    U1 = (sign(u - α) - 1) / (2 * gamma(1 + 1/p))
    U2 = (sign(u - α) + 1) / (2 * gamma(1 + 1/p))
    Y = α * W1^(1/p) * U1 + (1-α) * W2^(1/p) * U2
    σ*Y + μ
end

function loglik(θ, p, x)
    μ, σ, α = θ
    σ = exp(σ)
    -log.(pdf.(Aepd(μ, σ, p, α), x)) |> sum
end

function MLE(θ::Array{T, 1}, p::T, x::Array{T, 1}) where {T <: Real}
    length(θ) === 3 || throw(ArgumentError("θ not of length 3"))
    optimum = optimize(b -> loglik(b, p, x), θ, BFGS())
    Optim.converged(optimum) || @warn("Not converged")
    mle = Optim.minimizer(optimum)
    mle[2] = exp(mle[2])
    mle
end

end
