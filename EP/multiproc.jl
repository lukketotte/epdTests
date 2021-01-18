# using Distributed
@everywhere include(joinpath(@__DIR__, "ep.jl"))
@everywhere include(joinpath(@__DIR__, "NormTest.jl"))
@everywhere using .NormTest,.EPmethods
@everywhere using SpecialFunctions, Statistics, LinearAlgebra, Distributions
@everywhere using DataFrames, CSV, DataFramesMeta

using KernelDensity, Plots, PlotThemes, LaTeXStrings
theme(:juno)

## Simulation
@everywhere function simulation(p::Array{T, 1}, n::Array{N, 1}, nsim::N;
    α::T = 0.05, twoSided = true) where {T, N <: Real}
    simData = DataFrame(n = repeat(n, inner = length(p)), p = repeat(p, length(n)), value = 0.)
    for i in 1:length(n)
        print(string(n[i])*" ")
        for j in 1:length(p)
            res = zeros(10)
            for rep in 1:10
                q = pmap(N -> simSize(Epd(0.0, 1.0, p[j]), n[i], N, p[j]; twoSided = true, α = α), [nsim for x in 1:5])
                res[i] = mean(q)
            end
            simData[:, [:value]] .= ifelse.((simData[!, :n] .== n[i]) .& (simData[!, :p] .== p[j]), mean(res), simData[:, [:value]])
        end
    end
    simData
end

@everywhere function simulation(p::T, n::Array{N, 1}, nsim::N, gridSize::N;
        gridEnd::T = 5., α::T = 0.05, twoSided = true) where {T, N <: Real}

    pGrid = range(p, gridEnd, length = gridSize)
    simData = DataFrame(n = repeat(n, inner = gridSize), p = repeat(pGrid, length(n)), p0 = p, value = 0.)

    for i in 1:length(n)
        print(string(n[i])*" ")
        res = pmap(kurt -> simSize(Epd(0., 1., kurt), n[i], nsim, p; twoSided = twoSided, α = α), pGrid)
        simData[simData.n .== n[i], :value] = res
    end
    simData, pGrid
end


# Based on micheaux
@everywhere function simulation(n::Array{N, 1}, nsim::N, gridSize::N;
        gridEnd::T = 5., α::T = 0.05, twoSided = true) where {T, N <: Real}

    pGrid = range(2., gridEnd, length = gridSize)
    simData = DataFrame(n = repeat(n, inner = gridSize), p = repeat(pGrid, length(n)), p0 = 2., value = 0.)

    for i in 1:length(n)
        print(string(n[i])*" ")
        res = pmap(kurt -> simSize(n[i], nsim, kurt; twoSided = twoSided, α = α), pGrid)
        simData[simData.n .== n[i], :value] = res
    end
    simData, pGrid
end

## Calculation checks
NormTest.components(1., 1., 0.)

## testing
# estimated critical values for n = 20, 50, 100, 200
αLapGel = [0.51, 0.235, 0.135, 0.083]
αLap = [0.0625, 0.057, 0.053, 0.051]
simSizeLaplace(Laplace(0.), 200, 100000, true, quantile(Chisq(1), 1-0.051))
simSizeLaplace(Laplace(0.), 200, 30000, false, quantile(Chisq(1), 1-0.083); C₂ = 1400.)

simSizeLaplace(Cauchy(), 100, 100000, true, quantile(Chisq(1), 1-αLap[3]))
simSizeLaplace(Cauchy(), 50, 100000, false, quantile(Chisq(1), 1-αLapGel[2]))

simSize(Epd(0., 1., 1.), 500, 1000, 1.)


simSizeLaplace(Epd(0., 1., 1.), 1000, 5000, true, quantile(Normal(), 1-0.05))

simSize(TDist(3), 100, 1000, true)

simulation([50, 100], 100, 10; twoSided = false)

# EPD, MC estimated critical values
αNorm = [0.1, 0.069, 0.059, 0.053]
αNormMitch = [0.054, 0.053, 0.05, 0.05]
pmap(nsim -> simSize(Epd(0., 1., 2.), 200, nsim, true; α = 0.053), repeat([10000], 5)) |> mean
pmap(nsim -> simSize(Epd(0., 1., 2.), 100, nsim, false; α = 0.05), repeat([10000], 5)) |> mean

##  size
sim2 = simulation([1., 2., 3.], [50, 100], 500; α = 0.01, twoSided = false)
CSV.write("simsize_01.csv", sim2)

sim3 = simulation([1., 2., 3.], [50, 100, 500], 2500; α = 0.05, twoSided = false)
CSV.write("simsize_01.csv", sim3)

@linq sim3 |>
    where(:p .== 3)


## Simulate power
β₁, grid₁  = simulation(1., [50, 100, 500], 10000, 20; gridEnd = 4., α = 0.01, twoSided = false)
β₂, grid₂ = simulation(2., [50, 100, 500], 100, 20; gridEnd = 5., α = 0.05, twoSided = false)
β₃, grid₃  = simulation(3., [50, 100, 500], 10000, 20; gridEnd = 6., α = 0.01, twoSided = false)

βₘ, grid = simulation([50, 100, 500], 100, 20; twoSided = false)

CSV.write("power001.csv", [β₁; β₂; β₃])
# CSV.write("powerM.csv", βₘ)

## Comparison with Laplace test
# EPD
N, nsim = [100, 200], 50000
p = range(0.5, 2, length = 10)
simDat = DataFrame(n = repeat(N, inner = length(p)), p = repeat(p, length(N)), value = 0.0)
β = pmap(kurt -> simSizeLaplace(Epd(0., 1., kurt), N[2], 50000, false, quantile(Chisq(1), 1-αLapGel[4])), p)
simDat[simDat.n .== N[2], :value] = β

CSV.write("powerLapGel.csv", simDat)
# EPD for n = 100, 200

#Tdist
N, nsim = [20, 50, 100, 200], 10000
α = [0.04, 0.044, 0.048, 0.05]
α = [0.01, 0.024, 0.033, 0.04]
ν = [1, 2, 3, 4, 5, 6, 7]
simDat = DataFrame(n = repeat(N, inner = length(ν)), df = repeat(ν, length(N)), value = 0.0)
for i in 1:length(N)
    β = pmap(df -> simSizeLaplace(TDist(df), N[i], 50000, true, quantile(Chisq(1), 1-αLap[i])), ν)
    simDat[simDat.n .== N[i], :value] = β
end
CSV.write("powerLapTOurs.csv", simDat)

simSizeLaplace(Cauchy(), 50, 10000, true, quantile(Chisq(1), 1-0.044))
simSizeLaplace(Cauchy(), 100, 10000, false, quantile(Chisq(1), 1-0.024))

## comparison with T dist
N, nsim = [20, 50, 100, 200], 50000
ν = [1, 2, 3, 4, 5, 7, 10, 15, 20]
simDat = DataFrame(n = repeat(N, inner = length(ν)), df = repeat(ν, length(N)), value = 0.0)
for i in 1:length(N)
    β = pmap(df -> simSize(TDist(df), N[i], nsim, true; α = αNorm[i]), ν)
    simDat[simDat.n .== N[i], :value] = β
end
CSV.write("powerNormOursT.csv", simDat)

## comparison using the EPD
p  = range(1., 4, length = 20)
simDat = DataFrame(n = repeat(N, inner = length(p)), p = repeat(p, length(N)), value = 0.0)

for i in 1:length(N)
    β = pmap(kurt -> simSize(Epd(0.0, 1.0, kurt), N[i], nsim, false; α = αNormMitch[i]), p)
    simDat[simDat.n .== N[i], :value] = β
end

CSV.write("powerMich.csv", simDat)

#simData[simData.n .== n[i], :value] = res
simDat[simDat.n .== 20, :value] = powerTm

powerTm = pmap(df -> simSize(TDist(df), n, nsim, true), ν)
powerTm = pmap(df -> simSize(TDist(df), n, nsim, false), ν)

n, nsim = 500, 1500;
kurt = range(1, 3, length = 30)
β₀ = pmap(p -> αPar(p, 50, nsim, true), kurt)
β₁ = pmap(p -> αPar(p, 100, nsim, true), kurt)
β₃ = pmap(p -> αPar(p, 200, nsim, true), kurt)
β₂ = pmap(p -> αPar(p, 500, nsim, true), kurt)

α₀ = pmap(n -> αPar(2., n, nsim, true), [25, 50, 100, 250])
println(α₀)

α = pmap(n -> αPar(2., n, nsim, true, [0, log(2), 0.5]), [25, 50, 100, 250])
println(α)

α = pmap(N -> αPar(2., 5000, N, true, [0, log(2), 0.5]), [1000 for x in 1:4])
mean(α)
