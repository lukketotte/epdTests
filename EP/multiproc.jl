@everywhere include(joinpath(@__DIR__, "ep.jl"))
@everywhere include(joinpath(@__DIR__, "aep.jl"))
@everywhere include(joinpath(@__DIR__, "NormTest.jl"))
@everywhere using .NormTest, .AEPmethods, .EPmethods, SpecialFunctions, Statistics, LinearAlgebra, Distributions
@everywhere using DataFrames, CSV, DataFramesMeta

using KernelDensity, Plots, PlotThemes, LaTeXStrings
theme(:juno)

## Power and size
# TODO: have always assumed it to be 2. Need to fix
@everywhere function αPar(p::T, n::N = 50, nsim::N = 10000,
    size::Bool = false, α::T = 0.05) where {T, N <: Real}
    sim = simSize(Epd(0.0, 1.0, p), n, nsim, size, α, p)
    if size
        mean(sim)
    else
        sim
    end
end

@everywhere function αPar(p::T, θ::Array{T, 1}, n::N = 50,
    nsim::N = 10000, size::Bool = false, α::T = 0.05) where {T, N <: Real}
    sim = simSize(Aepd(0.0, 1.0, p, θ[3]), n, nsim, θ, size, α)
    if size
        mean(sim)
    else
        sim
    end
end

## Simulation
# TODO: simsize function has been changed, and αPar should not be used
@everywhere function simulation(p::Array{T, 1}, n::Array{N, 1}, nsim::N;
    α::T = 0.05, twoSided = true) where {T, N <: Real}
    simData = DataFrame(n = repeat(n, inner = length(p)), p = repeat(p, length(n)), value = 0.)
    for i in 1:length(n)
        print(string(n[i])*" ")
        for j in 1:length(p)
            res = Array{Float64, 1}[]
            for rep in 1:10
                q = pmap(N -> αPar(p[j], n[i], N, false), [nsim for x in 1:5])
                push!(res, reduce(vcat, q))
            end
            if twoSided
                z = quantile(Normal(), 1-α/2)
                resAvg = map(k -> mean(abs.(res[k]) .> z), 1:10) |> mean
            else
                z = quantile(Normal(), 1-α)
                resAvg = map(k -> mean((res[k]) .> z), 1:10) |> mean
            end
            simData[:, [:value]] .= ifelse.((simData[!, :n] .== n[i]) .& (simData[!, :p] .== p[j]), mean(resAvg), simData[:, [:value]])
        end
    end
    simData
end

# β₀ = pmap(p -> αPar(p, 50, nsim, true), kurt)
kurt = range(2, 5, length = 30)
pmap(p -> αPar(p, 50, nsim, true), kurt)

# that is working
pmap(p -> simSize(Epd(0, 1, p), 100, 1000, true, 0.05, 2.), kurt)

# this one is just giving the size now?
αPar(5., 1000, 100, true)

simSize(Epd(0.0, 1.0, 2.), 100, 100, true, 0.05, 2.)

@everywhere function simulation(p::T, n::Array{N, 1}, nsim::N, gridSize::N;
        gridEnd::T = 5., size::Bool = false, α::T = 0.05, twoSided = true) where {T, N <: Real}

    pGrid = range(p, gridEnd, length = gridSize)
    simData = DataFrame(n = repeat(n, inner = gridSize), p = repeat(pGrid, length(n)), p0 = p, value = 0.)

    for i in 1:length(n)
        print(string(n[i])*" ")
        res = Array{Float64, 1}[]
        for rep in 1:10
            q = pmap(N -> αPar(p, n[i], N, false), [nsim for x in 1:5])
            push!(res, reduce(vcat, q))
        end
        if twoSided
            z = quantile(Normal(), 1-α/2)
            resAvg = map(k -> mean(abs.(res[k]) .> z), 1:10) |> mean
        else
            z = quantile(Normal(), 1-α)
            resAvg = map(k -> mean((res[k]) .> z), 1:10) |> mean
        end
        simData[:, [:value]] .= ifelse.(simData[!, :n] .== n[i], mean(resAvg), simData[:, [:value]])
    end
    simData
end

test = simulation(1., [50, 100], 100, 30)
@linq test |>
    where(:n .== 50)


pgrid = range(2, 5, length = 5)
dat1 = DataFrame(n = repeat([50, 100], inner = 5), p = repeat(pgrid, 2), p₀ = 2)
pgrid = range(1, 5, length = 5)
dat2 = DataFrame(n = repeat([50, 100], inner = 5), p = repeat(pgrid, 2), p₀ = 1)

[dat1; dat2]

###


sim2 = simulation([1., 2., 3.], [50, 100], 500; α = 0.01, twoSided = false)
CSV.write("simsize_01.csv", sim2)

sim3 = simulation([1., 2., 3.], [50, 100, 500], 2500; α = 0.05, twoSided = false)
CSV.write("simsize_01.csv", sim3)

@linq sim3 |>
    where(:p .== 3)


## Simulate power


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

## power
p = plot(kurt, β₀, label = "n=50", xlab="p",
    ylab="power", legend=:outertopright, lw = 2.5, fontfamily=font("sans-serif"),
    legendfontsize=11, xtickfontsize=11, ytickfontsize=10,
    linewidth = 1.5, grid = false)
p = plot!(kurt, β₁, label = "n=100", xlab="p", ylab="power", lw = 2.5, linestyle= :dot)
p = plot!(kurt, β₃, label = "n=200", xlab="p", ylab="power", lw = 2.5, linestyle=:dashdotdot)
p = plot!(kurt, β₂, label = "n=500", xlab="p", ylab="power", lw = 2.5, linestyle=:dashdot)
p = vline!([2], linestyle=:dash, color = "gray", label = "")
p = plot!(size=(700,400), dpi = 500)
savefig(p, "power2.png")

# fg_legend = :transparent,background_color_legend = :transparent
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

π/2
p = 2
2*p^(1/p) * gamma(1 + 1/p)

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



n, nsim = 4000, 2500
# q = pmap(N -> αPar(2., [0, log(2), 0.5], n, N, false), [1000 for x in 1:4])
res = Array{Float64, 1}[]
for i in 1:10
    print(string(i)*" ")
    q = pmap(N -> αPar(2., n, N, false), [nsim for x in 1:4])
    push!(res, vcat(q[1], q[2], q[3], q[4]))
end

res = map(i -> mean(abs.(res[i]) .> 1.96), 1:10)
mean(res)
push!(res, vcat(q))
Array(q)
n, nsim = 500, 10
res = Array{Float64, 1}[]
for i in 1:10
    print(string(i)*" ")
    q = pmap(N -> αPar(1., n, N, false, 0.01), [nsim for x in 1:4])
    push!(res, vcat(q[1], q[2], q[3], q[4]))
end

reduce(hcat, q)
reduce(vcat, q)

z = quantile(Normal(), 1-0.01/2)
resAvg = map(i -> mean(abs.(res[i]) .> z), 1:10)
mean(resAvg) |> println

q
convert(Array{Float64, 1}, q)
Array{Float64, 1}
n, nsim = 500, 1000
res = Array{Float64, 1}[]
for i in 1:10
    print(string(i)*" ")
    q = pmap(N -> αPar(2., [0, log(2), 0.7], n, N, false), [nsim for x in 1:4])
    push!(res, vcat(q[1], q[2], q[3], q[4]))
end

z = quantile(Normal(), 1-0.01/2)
resAvg = map(i -> mean(abs.(res[i]) .> z), 1:10)
mean(resAvg) |> println


qq = vcat(q[1], q[2], q[3], q[4])
(abs.(qq) .> 1.96) |> mean

qqPlot(qq, Normal(0,1), "QQ plot, n = " * string(n))
k, x = kde(qq), range(-4, 4, length = 500);
plot(x, pdf(k, x), label = "aepd")
plot!(x, pdf.(Normal(), x))


k, x = kde(qq.^2), range(0, 6, length = 500);
plot(x, pdf(k, x))
plot!(x, pdf.(Chisq(1), x))

qqPlot(qq.^2, Chisq(1), "n = " * string(n))
