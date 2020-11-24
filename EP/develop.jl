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

NormTest.components(p, σ, μ)
NormTest.components(p, σ, μ, 0.9)

## size

rand(Aepd(0, 1, 2, 0.5), 1000) |> var
rand(Epd(0, 1, 2), 1000) |> var

x = rand(Aepd(0, 1, 2, 0.5), 1000);
x = rand(Epd(0, 1, 2), 1000);
test(x, 0., √(1. * π/2))
test(x, 0., √(1 * π/2), 0.5)

function simSizes(d::Aepd, n::N, nsim::N, size::Bool = true,
    θ::Array{T, 1} = ones(3)) where {N <: Integer, T <: Real}
    sims = [0. for x in 1:nsim]
    for i in 1:nsim
        y = rand(d, n)
        μ, σ, α = try
            MLE(θ, 2., y)
        catch err
            NaN, NaN, NaN
        end
        if μ === NaN
            sims[i] = NaN
        else
            t = test(y, μ, σ * √(π/2), α)
            if size
                if abs(t) > 1.96
                    sims[i] = 1
                end
            else
                sims[i] = t
            end
        end
    end
    sims[sims .!== NaN]
end

function simSizes(d::Epd, n::N, nsim::N, size::Bool = true) where {N <: Integer}
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

sims = simSizes(Epd(0, 1, 2), 1000, 1000) |> mean

sims = simSizes(Aepd(0, 1, 2, 0.5), 500, 1500, true, [0, log(2), 0.5])
mean(sims)
