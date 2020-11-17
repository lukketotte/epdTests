include("..\\EP\\ep.jl")
include("..\\EP\\aep.jl")

using .EPmethods, .AEPmethods, Plots, PlotThemes, Distributions, Formatting, LaTeXStrings

## Display the AEPD
x1 = range(-7, 3, length = 500)
x2 = range(-3, 7, length = 500)

kurt = [1, 2, 5]
α1, α2 = 0.8, 0.2

data_1 = [pdf.(Aepd(0, 1, p, α1), x1) for p in kurt]
data_2 = [pdf.(Aepd(0, 1, p, α2), x2) for p in kurt]

feKurt = FormatExpr("p = {1}")
labels = [format(feKurt, p) for p in kurt] |> (y -> reshape(y, 1,3));
linestyles = [:solid :dash :dashdot];
feAlpha = FormatExpr(L"\alpha = {1}")

p1 = plot(x1, data_1, label = labels, color = "black",
    linestyle = linestyles, grid = false, legend = false,
    title = format(feAlpha, α1));

p2 = plot(x2, data_2, label = labels, color = "black",
    linestyle = linestyles, grid = false, legend = true,
    title = format(feAlpha, α2));

plot(p2, p1, layout = (1,2))
p = plot!(size = (500, 250))
savefig(p, "aepd.png")

x = range(-4, 4, length = 500)
data = [pdf.(Epd(0, 1, p), x) for p in kurt]
p = plot(x, data, label = labels, color = "black",
    linestyle = linestyles, grid = false, size = (300, 250))
savefig(p, "epd.png")
