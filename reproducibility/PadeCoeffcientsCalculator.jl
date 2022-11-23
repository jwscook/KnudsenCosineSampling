using LsqFit, Roots, Plots, Combinatorics, QuadGK, SpecialFunctions
using BlackBoxOptim, LinearAlgebra

const Δ = 1e8eps()

function invf(f, x)
 return try
     Roots.find_zero(y->f(y) - x, 0.5)
 catch
   @warn "Retrying root find with $x"
   Roots.find_zero(y->f(y) - x, 0.5, Secant())
 end
end

#padedivide(n::Int) = 3n ÷ 4 + (iseven(n) ? 1 : 0)
padedivide(n::Int) = n ÷ 2 + (iseven(n) ? 0 : 1)

function padeab(p)
 n = length(p)
 m = padedivide(n)
 as = @inbounds @view p[1:m]
 bs = @inbounds @view p[m+1:n]
 return (as, bs)
end

function pade(x::T, p, transformop::F)::T where {T,F}
#  x = sqrt(-log(sqrt(1-x))) # tan(x * π/2) #
#  x = cbrt(-log((1-x))) # tan(x * π/2) #
  x = transformop(x)
  as, bs = padeab(p)
  num = evalpoly(x, as) * x #sum(as[i] * x^i for i in eachindex(as))
  denom = 1 + evalpoly(x, bs) * x #sum(bs[i] * x^i for i in eachindex(bs))
  denom == 0 && return zero(T) # allow the optimisation to proceed
  return T(num / denom)
end

pades(x::Number, p, ncoeffs, transformop::F) where F = pade(x, p, transformop)

pades(x, p, ncoeff, transformop::F) where F = [pades(xi, p, ncoeff, transformop) for xi in x]

function quadratureobjective(p, ncoeff, f::F, transformop::G; rtol=1e-12) where {F,G}
  return quadgk(x->abs(pades(x, p, ncoeff, transformop) - invf(f, x)), 0, 1-Δ, rtol=rtol)[1]
end

function quadratureoptimisation(objective::F, ncoeff, guess, transformop::G; timelimitperncoeff=10) where {F,G}
  inner(p) = 1e6 * quadratureobjective(p, ncoeff, objective, transformop)
#  searchrange = (-1e1, 1e1)
  searchrange = (-1e3, 1e3)
#  searchrange = [(g .- abs(g)/10, g .+ abs(g) / 10) for g in guess]
  res = bboptimize(inner, guess; SearchRange=searchrange, NumDimensions=length(guess),
    MaxSteps=1000000000, TraceInterval=10, MaxTime=timelimitperncoeff * ncoeff,
    Method=:adaptive_de_rand_1_bin_radiuslimited, TraceMode=:silent)#:adaptive_de_rand_1_bin
  coeffs = best_candidate(res)
  fitness = best_fitness(res)
  @show fitness, coeffs
  return x->pades(x, coeffs, ncoeff, transformop), coeffs
end

function discretefit(objective::F, ncoeff, transformop::G, c = collect(1/200:1/100:1-1/200)) where {F,G}
  function m(x, p)
    @assert length(p) == ncoeff
    return pades(x, p, ncoeff, transformop)
  end
  fit = curve_fit(m, c, invf.(objective, c), ones(ncoeff))
  return x->m(x, fit.param), fit.param
end

function calculatefit(f::F, fname, transformop::G;
    N=1000, ncoeffs=(8, 12, 16), timelimitperncoeff=10) where {F, G}
  ncoeffs = collect(ncoeffs)
  c = collect(1/2N:1/N:1 - 1/N/2)
  y = invf.(f, c)
  d = Dict()
  h = Plots.plot()
  sort!(ncoeffs)
  for ncoeff in ncoeffs
    try
      println("")
      @show fname, ncoeff
      guessfit, guessparams = discretefit(f, ncoeff, transformop, c)
      fit, param = quadratureoptimisation(f, ncoeff, guessparams, transformop; timelimitperncoeff=timelimitperncoeff)
      z = [fit(ci) for ci in c]
      @assert fit(eps()) > 0
      @assert fit(1 - eps()) > 0
      @assert all(z .> 0)
      err, errerr = QuadGK.quadgk(x->abs(fit(x) - invf(f, x)), 0, 1 - Δ)
  #    gradf = ForwardDiff.gradient(x->quadratureobjective(x, ncoeff, f, transformop), param)
  #    hessf = Calculus.hessian(x->quadratureobjective(x, ncoeff, f, transformop), param)
  #    mse = quadratureobjective(param, ncoeff, f, transformop, rtol=1e-4)
  #    paramwitherrors = measurement.(param, sqrt.(mse * abs.(diag(inv(hessf)))))
  #    paramwitherrors = measurement.(param, param .* eps() ./ gradf)
      #@show paramwitherrors
      d[ncoeff] = (param, err, fit, )#paramwitherrors)
      Plots.plot!(h, c, z .- y, label="$ncoeff, $err")
    catch err
      @show err
    end
  end
  try
    println("\n$fname \n")
    bestncoeff = collect(keys(d))[findmin([d[k][2] for k in keys(d)])[2]]
    line = " & {$bestncoeff coefficients, error $(round(d[bestncoeff][2], sigdigits=1))}"
    line *= "\\\\"
    println(line)
    println("\\midrule \n i & a_i & b_i \\\\ \n \\midrule")
    coeffs = d[bestncoeff][1]
    as, bs = padeab(coeffs)
    for i in 1:max(length(as), length(bs))
      sa = i <= length(as) ? "$(as[i])" : "-"
      sb = i <= length(bs) ? "$(bs[i])" : "-"
      line = "$i & $sa & $sb \\\\"
      println(line)
    end

    savefig(h, fname * ".pdf")
    return (h, d)
  catch err
    @show err
    return nothing
  end

end

funi(x) = erf(x) - 2 / sqrt(pi) * x * exp(-x^2) # 12
fv(x) = (y = -x^2; return y * exp(y) - expm1(y)) # 21 ## 1 - (1 + x^2) * exp(-x^2) # 21
fvx(x) = sqrt(pi) * x^3 * (1-erf(x)) - x^2 * exp(-x^2) - expm1(-x^2) # 31 # sqrt(pi) * x^3 * (1-erf(x)) -(1 + x^2) * exp(-x^2) + 1 # 31
fv⊥(x) = erf(x) * (1 - 2 * x^2) + 2x * (x - exp(-x^2)/sqrt(pi)) # 32

defaulttransformop(x) = sqrt(-log(1-x))
cbrttransformop(x) = cbrt(-log(1-x))
unitransformop(x) = sqrt(-log(1-cbrt(x)^2))
vxtransformop(x) = cbrt(-log(1-sqrt(x)))^2
vperptransformop(x) = (-log(1-x))^(5/12)
lamberttransformop(x) = sqrt(-1 - lambertw((x - 1) / exp(1), -1))
othertransform(x) = sqrt(-log(1-sqrt(x)))
quarterroot(x) = sqrt(sqrt(-log(1-x)))

tlim=30
#h = calculatefit(funi, "Uniform_v cbrttransformop", cbrttransformop; timelimitperncoeff=tlim)
#h = calculatefit(fv, "Wedge_v cbrttransformop", cbrttransformop; timelimitperncoeff=tlim)
h = calculatefit(fv, "Wedge_v quarterroot", quarterroot; timelimitperncoeff=tlim)
#h = calculatefit(fv, "Wedge_v othertransform", othertransform; timelimitperncoeff=tlim)
#h = calculatefit(fvx, "Wedge_vx defaulttransformop", defaulttransformop; timelimitperncoeff=tlim)
#h = calculatefit(fvx, "Wedge_vx vxtransformop", vxtransformop; timelimitperncoeff=tlim)
#h = calculatefit(fvx, "Wedge_vx cbrttransformop", cbrttransformop; timelimitperncoeff=tlim)
#h = calculatefit(fv⊥, "Wedge_vperp defaulttransformop", defaulttransformop; timelimitperncoeff=tlim)
#h = calculatefit(fv⊥, "Wedge_vperp cbrttransformop", cbrttransformop; timelimitperncoeff=tlim)



