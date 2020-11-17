## Code for the simplified test that Rauf wanted me to investigate
include("..\\EP\\ep.jl")
include("..\\EP\\NormTest.jl")
using .NormTest, .EPmethods, Distributions, Plots, PlotThemes, KernelDensity, SpecialFunctions

y = rand(Epd(0, 1, 2), 100)

function t(x, μ, σ)
    (x .- μ) ./ σ
end

(t(y, 0, 1)).^2 |> mean

nSim, n = 2000, 5000
sims = [0. for x in 1:nSim]

for i in 1:nSim
    y = rand(Normal(0, 1), n);
    sims[i] = √n * (abs.(t(y, mean(y), √(var(y) * π/2))) |> mean)
end

for i in 1:nSim
    y = rand(Normal(0, 1), n)
    est = mean(abs.(y .- mean(y)))
    sims[i] = √n * (est - √(2/π))
end

k = kde(sims);
x = range(minimum(sims)-0.5, maximum(sims) + 0.5, length = 500);
plot(x, pdf(k, x))
plot!(x, pdf.(Normal(0, √(1-2/π)), x))
