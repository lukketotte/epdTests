using QuadGK, SpecialFunctions, Plots, PlotThemes
theme(:juno)

## Testing integral in (49) and (50) of Walsh & Zhu
function f(x::T, r::T = -.95, m::T = .0, p::T =.5) where {T <: Real}
    x^((1+r)/p - 1) * (log(x))^m * exp(-x)
end

quadgk(x-> f(x, 2.0 - 1, 1., 2.), 0, Inf)
digamma(1)/(2*gamma(1 + 1/2))

## testing similar in Agró
function g(x, r=-1/2, σ = 1., p=1., μ=0.)
    abs(x-μ)^r * exp(-abs(x-μ)^p / (p*σ^p)) / (2*p^(1/p) * σ * gamma(1 + 1/p))
end

quadgk(x -> g(x, r, 1., p, 0.), -Inf, Inf)
p^(r/p) * gamma((r+1)/p)/gamma(1/p)


function g(x::T, μ::T = 1., σ::T = 2., p::T = 1.5) where {T <: Real}
    # a = 1/(σ * √(2*pi))
    # a *= exp(-.5 * ((x-μ)/σ)^2)
    a = 1/(2 * p^(1/p) * σ * gamma(1 + 1/p))
    a *= exp(-1/(p*σ) * abs(x-μ)^p)
    b = 2/p^3 * (log(p) - 1 + digamma(1 + 1/p) - p * (abs(x-μ)/σ)^p *log(abs(x-μ)/σ) + (abs(x-μ)/σ)^p)
    b -= 1/p^2 *(1/p - 1/p^2 * trigamma(1 + 1/p) - p*(abs(x-μ)/σ)^p * log(abs(x-μ)/σ)^2)
    a*b
end

quadgk(g, -Inf, Inf, rtol=1e-5)


function Kₑ(p::T) where {T <: Real}
    1/(2*p^(1/p) * gamma(1 + 1/p))
end

function EP(x::T, μ::T = 0., σ::T = 1., p::T = 2.; scaled::Bool = true) where {T <: Real}
    k = Kₑ(p)
    if scaled
        e = exp(-1/p * abs((x-μ)/(Kₑ(p)*σ))^p)
        e/(σ)
    else
        e = exp(-1/p * abs((x-μ)/σ)^p)
        e * Kₑ(p) / σ
    end
end

EP(0.5, scaled = true)
EP(0.5/Kₑ(2), scaled = false)

x = range(-5, 5, length = 200);
plot(x, EP.(x, scaled = true))
plot!(x, EP.(x, scaled = false))


## checks

a = 0.5 * (1/4 * (digamma(3/2) + trigamma(3/2)) + digamma(3/2))
a += 2 * digamma(1) / (4*gamma(3/2)) - digamma(3/2) * digamma(1)/(4*gamma(3/2))

1/8 * (1 + 1/2) * digamma(3/2)
