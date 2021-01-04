include("..\\EP\\ep.jl")
include("..\\EP\\aep.jl")

using .EPmethods, .AEPmethods, Plots, PlotThemes, Distributions, Formatting, LaTeXStrings, SpecialFunctions


z = quantile(Normal(), 1-0.05/2)
I = 3/16 * trigamma(3/2)
cdf(Normal(), I^0.5)

h = range(0, 20, length = 2000)

p = plot(h, 1 .- cdf.(Normal(), z .- h.*I^0.5), lw = 1.5, label = L"\sigma", xlab = L"h_p", ylab = "power")
p = plot!(h, 1 .- cdf.(Normal(), z .- h.*(I-1/8)^0.5), lw = 1.5, label = L"\hat{\sigma}")
p = plot!(size = (700, 400), legendfontsize=11,
    xtickfontsize=10, ytickfontsize=10, guidefontsize=12, legend=:bottomright)
p = plot!(size=(500,300), dpi = 500)
savefig(p, "powercomp.png")

I^0.5
(I-1/8)^0.5

cdf(Normal(), I^0.5)
cdf(Normal(), (I-1/8)^0.5)
