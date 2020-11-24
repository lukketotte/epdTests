module EPmethods

export Epd

using Distributions, SpecialFunctions, Random
import Distributions.pdf, Distributions.quantile, Base.rand


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

end
