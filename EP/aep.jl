module AEPmethods

export Aepd

using Distributions, SpecialFunctions, Random
import Distributions.pdf, Distributions.quantile, Base.rand

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

end
