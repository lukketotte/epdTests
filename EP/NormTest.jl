module NormTest

export test, α, L, S, IM

include("ep.jl")
using .EPmethods, SpecialFunctions, Statistics, LinearAlgebra, Distributions

function L(y::T, μ::T, σ::T) where {T <: Real}
    gamma(3/2) * abs(y - μ) / σ
end

"""
Computes the score function wrt p & σ
"""
function S(y::T, μ::T, σ::T) where{T <: Real}
    ℓ = L(y, μ, σ)
    S_p = ℓ^2*(1/2 * digamma(3/2) - log(ℓ))
    S_σ = 2/σ * ℓ^2 - 1/σ
    return S_p, S_σ
end

"""
Computes the score function wrt p & σ for a vector y
"""
function S(y::Array{T, 1}, μ::T, σ::T) where{T <: Real}
    ℓ = L.(y, μ, σ)
    S_p = ℓ.^2 .* (1/2 * digamma(3/2) .- log.(ℓ))
    S_σ = 2/σ .* ℓ.^2 .- 1/σ
    return S_p, S_σ
end

"""
Computes the components from the information matrix
"""
function components(n::N, p::T, σ::T, μ::T) where {T <: Real, N <: Int}
    inf = IM(n, p, σ, μ)
    A, B = reshape(inf[1, 2:3], 2, 1), inv(inf[2:3, 2:3])
    β = (B * A)[2]
    V = inf[1, 1] - (transpose(A) * B * A |> first)
    β, V
end

"""
Computes the fisher information for the EP distribution
"""
function IM(n::N, p::T, σ::T, μ::T) where {T <: Real, N <: Int}
    ϕ₁₁ = p^(-3)*(1+1/p)*trigamma(1+1/p)
    ϕ₁₂, ϕ₂₃, ϕ₁₃ = 0, 0, -1/(σ*p)
    ϕ₂₂ = gamma(1/p)*gamma(2-1/p)/σ^2
    ϕ₃₃ = p/σ^2
    n.*[ϕ₁₁ ϕ₁₂ ϕ₁₃ ; ϕ₁₂ ϕ₂₂ ϕ₂₃; ϕ₁₃ ϕ₂₃ ϕ₃₃]
    [ϕ₁₁ ϕ₁₂ ϕ₁₃ ; ϕ₁₂ ϕ₂₂ ϕ₂₃; ϕ₁₃ ϕ₂₃ ϕ₃₃]
end

"""
Computes the C(α) test
"""
function test(y::Array{T, 1}, μ::T, σ::T) where {T <: Real}
    n = length(y)
    S_p, S_σ = S(y, μ, σ)
    β, V = components(n, 2., σ, μ)
    ((sum(S_p) - β * sum(S_σ)) / √(n*V))
end

"""
Computes empirical level of C(α) test
"""
function α(d::T, n::N, nsim::N, size::Bool = true) where {T <: ContinuousUnivariateDistribution, N <: Integer}
    sims = [0. for x in 1:nsim]
    for i in 1:nsim
        y = rand(d, n)
        t = test(y, mean(y), √(var(y) * (π/2)))
        if size
            if abs(t) > 1.96
                sims[i] = 1
            end
        else
            sims[i] = t
        end

    end
    sims
end
end
