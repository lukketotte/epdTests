@everywhere include(joinpath(@__DIR__, "ep.jl"))
@everywhere include(joinpath(@__DIR__, "aep.jl"))
@everywhere include(joinpath(@__DIR__, "NormTest.jl"))
@everywhere using .NormTest, .AEPmethods, .EPmethods
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


## testing
twoSided = false
α = 0.05
z = twoSided ? quantile(Normal(), 1-α/2) : quantile(Normal(), 1-α)
t = -2.1
(twoSided ? abs(t) : t) > z


# t-dist
quantile(Normal(), 1-0.05/2)
simSize(TDist(100), 20, 1000, true)
simSize(TDist(100), 20, 1000, false)

simulation([50, 100], 100, 10; twoSided = false)

# EPD
simSize(Epd(0., 1., 2.5), 500, 2000, true)
simSize(Epd(0., 1., 2.5), 500, 2000, false)

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

## comparison with T dist
N, nsim = [20, 50, 100, 200], 10000
ν = [1, 2, 3, 4, 5, 7, 10, 15, 20]
simDat = DataFrame(n = repeat(N, inner = length(ν)), df = repeat(ν, length(N)), value = 0.0)
for n in N
    β = pmap(df -> simSize(TDist(df), n, nsim, false), ν)
    simDat[simDat.n .== n, :value] = β
end
CSV.write("powerMich.csv", simDat)

## comparison using the EPD
p  = range(1., 4, length = 20)
simDat = DataFrame(n = repeat(N, inner = length(p)), p = repeat(p, length(N)), value = 0.0)

for n in N
    β = pmap(kurt -> simSize(Epd(0.0, 1.0, kurt), n, nsim, true), p)
    simDat[simDat.n .== n, :value] = β
end

CSV.write("powerCa.csv", simDat)

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
