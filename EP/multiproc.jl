# relative imports not working with everywhere include
@everywhere include(joinpath(@__DIR__, "ep.jl"))
@everywhere include(joinpath(@__DIR__, "aep.jl"))
@everywhere include(joinpath(@__DIR__, "NormTest.jl"))
@everywhere using .NormTest, .AEPmethods, .EPmethods, SpecialFunctions, Statistics, LinearAlgebra, Distributions, DataFrames

using KernelDensity, Plots, PlotThemes, LaTeXStrings
theme(:default)


## Power and size
@everywhere function αPar(p::T, n::N = 50, nsim::N = 10000, size::Bool = false, α::T = 0.05) where {T, N <: Real}
    sim = simSize(Epd(0.0, 1.0, p), n, nsim, size, α, p)
    if size
        mean(sim)
    else
        sim
    end
end

@everywhere function αPar(p::T, θ::Array{T, 1}, n::N = 50, nsim::N = 10000, size::Bool = false, α::T = 0.05) where {T, N <: Real}
    sim = simSize(Aepd(0.0, 1.0, p, θ[3]), n, nsim, θ, size, α)
    if size
        mean(sim)
    else
        sim
    end
end

## Simulation
res = Array{Float64, 1}[]
for i in 1:10
    print(string(i)*" ")
    q = pmap(N -> αPar(2., [0, log(2), 0.7], n, N, false), [nsim for x in 1:4])
    push!(res, vcat(q[1], q[2], q[3], q[4]))
end

@everywhere function simulation(p::Array{T, 1} = [1., 2., 3.], n::Array{N, 1} = [50, 100, 500, 1000, 2000],
        size::Bool = false, α::T = 0.05; nsim::N = 10000) where {T, N <: Real}
    simData = DataFrame(n = repeat(n, inner = length(p)), p = repeat(p, length(n)), value = 0.)
    for i in 1:length(n)
        print(string(n[i])*" ")
        for j in 1:length(p)
            res = Array{Float64, 1}[]
            for i in 1:10
                q = pmap(N -> αPar(p[j], n[i], N, false), [nsim for x in 1:4])
                push!(res, vcat(q[1], q[2], q[3], q[4]))
            end
            z = quantile(Normal(), 1-α/2)
            resAvg = map(i -> mean(abs.(res[i]) .> z), 1:10)
            test[:, [:value]] .= ifelse.((test[!, :n] .== n[i]) .&( test[!, :p] .== p[j]), mean(resAvg), test[:, [:value]])
        end
    end
    simData
end

simulation(nsim = 10)



p = [1., 2., 3.]
n = [50, 100, 500, 1000, 2000]

repeat(n, inner = 3)
repeat(p, 5)

test = DataFrame(n = repeat(n, inner = 3), p = repeat(p, 5), value = 0.)

test.value[1] = 0.1

test.value[(test.p .=== 1.) .& (test.n .=== 50) ] = 0.5

(test.p .=== 1.) .& (test.n .=== 50)

test[:, [:value]] .= ifelse.((test[!, :n] .== 50) .&( test[!, :p] .== 2.), 0.6, test[:, [:value]])



n, nsim = 100, 10000;
kurt = range(1, 5, length = 160)
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

n, nsim = 500, 10
res = Array{Float64, 1}[]
for i in 1:10
    print(string(i)*" ")
    q = pmap(N -> αPar(1., n, N, false, 0.01), [nsim for x in 1:4])
    push!(res, vcat(q[1], q[2], q[3], q[4]))
end

z = quantile(Normal(), 1-0.01/2)
resAvg = map(i -> mean(abs.(res[i]) .> z), 1:10)
mean(resAvg) |> println

#

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
