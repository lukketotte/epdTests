# relative imports not working with everywhere include
@everywhere include(joinpath(@__DIR__, "ep.jl"))
@everywhere include(joinpath(@__DIR__, "NormTest.jl"))
@everywhere using .NormTest, .EPmethods, SpecialFunctions, Statistics, LinearAlgebra, Distributions

using KernelDensity, Plots, PlotThemes, LaTeXStrings
theme(:default)

## Power and size

@everywhere function αPar(p::T, n::N = 50, nsim::N = 10000, size::Bool = false) where {T, N <: Real}
    sim = α(Epd(0.0, 1.0, p), n, nsim, size)
    if size
        mean(sim)
    else
        sim
    end
end

n, nsim = 100, 10000;
kurt = range(2, 5, length = 70)
β₀ = pmap(p -> αPar(p, 50, nsim, true), kurt)
β₁ = pmap(p -> αPar(p, 100, nsim, true), kurt)
β₂ = pmap(p -> αPar(p, 200, nsim, true), kurt)

α₀ = pmap(n -> αPar(2., n, nsim, true), [25, 50, 100, 250])
println(1 .- α₀)


## power
plot(kurt, 1 .- β₀, label = "n=50", xlab="kurtosis",
    ylab="power", legend=:topleft, lw = 2, fontfamily=font("sans-serif"))
plot!(kurt, 1 .- β₁, label = "n=100", xlab="kurtosis", ylab="power", lw = 2, linestyle= :dot)
plot!(kurt, 1 .- β₂, label = "n=200", xlab="kurtosis", ylab="power", lw = 2, linestyle=:dashdot)
plot!(size=(600,400))
# png("..\\latex\\figs\\power")

## Checking LLN
@everywhere function lln(n::N, nSim::N) where {N <: Integer}
    sim = [0. for x in 1:nSim]
    for i in 1:nSim
        y = rand(Epd(0,1,2), n)
        sim[i] = L.(y, mean(y), √(var(y) * (π/2))).^2
    end
    mean(sim)
end

y = rand(Epd(0,1,2), 100000)
(L.(y, mean(y), √(var(y) * (π/2))).^2) .* log.(L.(y, mean(y), √(var(y) * (π/2)))) |> mean

pmap(n -> lln(n, 1000), [2000 for x in 1:4]) |> mean

## QQ plot
function qqPlot(obs, F, title)
    nobs = length(obs)
    sort!(obs)
    quantiles = [quantile(F, i/nobs) for i in 1:nobs]
    plot(quantiles, obs, seriestype=:scatter, xlabel = "Theoretical quantiles",
         ylabel = "Sample Quantiles", title=title, label = "",
         markershape = :circle,
         markersize = 2)
    plot!(obs, obs, label = "")
end
n = 250
qq = αPar(2., n, 10000)
qqPlot(qq, Normal(0,1), "QQ plot, n = " * string(n))

qq = pmap(n -> αPar(2., n, 100), [25, 50, 100, 250])

qq[1]
