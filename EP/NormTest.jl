module NormTest

export test, simSize, L, R, S, IM, components

include("ep.jl")
include("aep.jl")
using .AEPmethods, .EPmethods, SpecialFunctions, Statistics, LinearAlgebra, Distributions


"""
L function for the EPD, under assumption p = 2
"""
function L(y::T, μ::T, σ::T; p::T = 2.) where {T <: Real}
    gamma(1 + 1/p) * abs(y - μ) / σ
end

"""
L function for the SEPD
"""
function L(y::T, μ::T, σ::T, α::T) where {T <: Real}
    y < μ ? L(y, μ, σ) / (2*α) : 0
end

"""
R function for the SEPD
"""
function R(y::T, μ::T, σ::T, α::T) where {T <: Real}
    y >= μ ? L(y, μ, σ) / (2*(1-α)) : 0
end


"""
Computes the score function wrt p & σ for the EPD
"""
function S(y::T, μ::T, σ::T; p::T = 2.) where{T <: Real}
    ℓ = L(y, μ, σ, p = p)
    S_p = ℓ^p * (1/p * digamma(1 + 1/p) - log(ℓ))
    S_σ = p/σ * ℓ^p - 1/σ
    return S_p, S_σ
end

"""
Computes the score functions wrt p & σ for the SEPD
"""
function S(y::T, μ::T, σ::T, α::T) where{T <: Real}
    l = L(y, μ, σ, α)
    r = R(y, μ, σ, α)
    c = digamma(3/2) / 2
    S_p = l == 0. ? r^2*(c-log(r)) : l^2*(c-log(l))
    S_σ = (2 * (l^2 + r^2) - 1) / σ
    return S_p, S_σ
end

"""
Computes the score function wrt p & σ for a vector y
"""
function S(y::Array{T, 1}, μ::T, σ::T; p::T = 2.) where{T <: Real}
    ℓ = L.(y, μ, σ, p = p)
    S_p = ℓ.^p .* (1/p * digamma(1 + 1/p) .- log.(ℓ))
    S_σ = p/σ .* ℓ.^p .- 1/σ
    return S_p, S_σ
end

"""
Computes the score functions wrt p & σ for the SEPD
"""
function S(y::Array{T, 1}, μ::T, σ::T, α::T) where{T <: Real}
    l = L.(y, μ, σ, α)
    r = R.(y, μ, σ, α)
    c = digamma(3/2) / 2
    n = length(y)
    S_p = similar(l)
    for i = 1:n
        S_p[i] = l[i] == 0. ?  r[i]^2*(c-log(r[i])) : l[i]^2*(c-log(l[i]))
    end
    S_σ = (2 .* (l.^2 .+ r.^2) .- 1) ./ σ
    return S_p, S_σ
end

"""
Computes the fisher information for the EP distribution
"""
function IM(p::T, σ::T, μ::T) where {T <: Real, N <: Int}
    ϕ₁₁ = p^(-3)*(1+1/p)*trigamma(1+1/p)
    ϕ₁₂, ϕ₂₃, ϕ₁₃ = 0, 0, -1/(σ*p)
    ϕ₂₂ = gamma(1/p)*gamma(2-1/p)/σ^2
    ϕ₃₃ = p/σ^2
    [ϕ₁₁ ϕ₁₂ ϕ₁₃ ; ϕ₁₂ ϕ₂₂ ϕ₂₃; ϕ₁₃ ϕ₂₃ ϕ₃₃]
end
"""
Computes the information matrix for the SEPD, θ = (p, μ, σ, α)
"""
function IM(p::T, σ::T, μ::T, α::T) where {T <: Real, N <: Int}
    # Diagonal elements
    ϕ_11 = (p + 1)/p^4 * trigamma(1 + 1/p)
    ϕ_22 = gamma(1/p) * gamma(2 - 1/p) / (σ^2 * α * (1-α))
    ϕ_33 = p/σ^2
    ϕ_44 = (p+1)/(α*(1-α))
    # Off-diags
    ϕ_12, ϕ_41, ϕ_32, ϕ_43 = 0, 0, 0, 0
    ϕ_13 = -1/(σ * p)
    ϕ_24 = -p/(σ * α*(1-α))
    # return array
    [ϕ_11 ϕ_12 ϕ_13 ϕ_41 ; ϕ_12 ϕ_22 ϕ_32 ϕ_24 ;
    ϕ_13 ϕ_32 ϕ_33 ϕ_43 ; ϕ_41 ϕ_24 ϕ_43 ϕ_44]
end

"""
Computes the components from the information matrix for the EPD
"""
function components(p::T, σ::T, μ::T) where {T <: Real, N <: Int}
    inf = IM(p, σ, μ)
    A, B = reshape(inf[1, 2:3], 2, 1), inv(inf[2:3, 2:3])
    β = (B * A)[2]
    V = inf[1, 1] - (transpose(A) * B * A |> first)
    β, V
end

"""
Computes the components from the information matrix for the SEPD
"""
function components(p::T, σ::T, μ::T, α::T) where {T <: Real, N <: Int}
    inf = IM(p, σ, μ, α)
    A, B = reshape(inf[1, 2:4],3,1), inv(inf[2:4, 2:4])
    β = (B * A)[2]
    V = inf[1, 1] - (transpose(A) * B * A |> first)
    β, V
end

"""
Computes the C(α) test for the EPD
"""
function test(y::Array{T, 1}, μ::T, σ::T; p::T = 2.) where {T <: Real}
    n = length(y)
    S_p, S_σ = S(y, μ, σ, p = p)
    β, V = components(p, σ, μ)
    ((sum(S_p) - β * sum(S_σ)) / √(n*V))
end

"""
Computes the C(α) test for the SEPD
"""
function test(y::Array{T, 1}, μ::T, σ::T, α::T) where {T <: Real}
    n = length(y)
    S_p, S_σ = S(y, μ, σ, α)
    β, V = components(2., σ, μ, α)
    ((sum(S_p) - β * sum(S_σ)) / √(n*V))
end

# TODO: why does e.g. d::Epd not work when using import?
"""
Computes empirical level of C(α) test for the EPD
"""
function simSize(d::D, n::N, nsim::N; twoSided::Bool = true, α::T = 0.05, p::T = 2.) where
    {D <: ContinuousUnivariateDistribution, N <: Integer, T<: Real}

    sims = [0. for x in 1:nsim]
    z = twoSided ? quantile(Normal(), 1-α/2) : quantile(Normal(), 1-α)

    for i in 1:nsim
        y = rand(d, n)
        if p == 1.
            μ = median(y)
            σ = mean(abs.(y .- μ))
        elseif p == 2.
            μ = mean(y)
            σ = √var(y)
        else
            μ, σ = try
                MleEpd([0, log(2.)], p, y)
            catch err
                NaN, NaN, NaN
            end
            σ !== NaN ? exp(σ) : NaN
        end

        if μ === NaN
            sims[i] = NaN
        else
            t = test(y, μ, p^(1/p) * gamma(1 + 1/p) * σ, p = p)
            if (twoSided ? abs(t) : t) > z
                sims[i] = 1
            end
        end
    end
    sims[sims .!== NaN] |> mean
end

"""
Computes empirical level of C(α) test for the SEPD
"""
function simSize(d::D, n::N, nsim::N, θ::Array{T, 1}, size::Bool = true, α::T = 0.05) where
    {D <: ContinuousUnivariateDistribution, N <: Integer, T <: Real}
    sims = [0. for x in 1:nsim]
    z = quantile(Normal(), 1-α/2)
    for i in 1:nsim
        y = rand(d, n)
        μ, σ, γ = try
            MLE(θ, 2., y)
        catch err
            NaN, NaN, NaN
        end
        if μ === NaN || (γ < 0 && γ > 1)
            sims[i] = NaN
        else
            t = test(y, μ, σ*√(π/2), γ)
            if size
                if abs(t) > z
                    sims[i] = 1
                end
            else
                sims[i] = t
            end
        end
    end
    sims[sims .!== NaN] |> mean
end

end
