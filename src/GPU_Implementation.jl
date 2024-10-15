using CUDA, KernelAbstractions, ForwardDiff, StatsBase, CairoMakie, PairPlots

"""
    pade(x::T,numer,denom) where T

Evaluate the Pade approximant at position x given numerator and denominator coefficients

...
# Arguments
- `x::T`: Evaluate the Pade approximant at x
- `numer`: the numerator coefficients
- `denom`: the denominator coefficients
...
"""
function pade(x::T, numer, denom) where T
  return T(evalpoly(x, numer) * x / (1 + evalpoly(x, denom) * x))
end

"""
    knudsencosine!(particles, Δt, vth)

Mutate the first argument, particles, initially comprising random numbers
so that it holds the values (x, vx, vy, vz) sampled from a Knudsen-cosine
distribution acting as a boundary source term, with timestep Δt and thermal
speed vth.

...
# Arguments
- `particles`: A 4xn matrix of particles quantities to be mutated in place.
Initially filled with random numbers, replaced with (x, vx, vy, vz)
- `Δt`: the duration of the timestep
- `vth`: the thermal speed characterising the Knudsen-cosine distribution
...
"""
@kernel function knudsencosine!(particles, Δt, vth)
    # Coefficients in Table 2
    numer = (1.1892074873464582, 2.3884028521993717, 0.021742384140688395,
             1.4551991003478761,−0.01545968501315156, 0.25068176822296523)
    denom = (2.0084051736325055, −0.2174611175058831, 0.7504356620416117,
             0.038026260588022796, 0.03376005807862799, −0.002372131900552844)
    f(x) = 1 - (1 + x^2) * exp(-x^2) # Eq. 23b
    j = @index(Global)
    # extract the random numbers from the matrix
    rx = particles[1, j]
    rv = particles[2, j]
    rθ = particles[3, j]
    rϕ = particles[4, j]
    # transformation operator g in Table 2
    z = sqrt(sqrt(-log(1 - rv)))
    # evaluate the Pade approximant
    v = pade(z, numer, denom)
    # perform a single Newton step if desired
    v -= (f(v) - rv) / ForwardDiff.derivative(f, v)
    # scale by thermal velocity
    v *= vth
    # evaluate the rest of the phase space positions
    θ = acos(cbrt(1 - rθ))
    ϕ = 2π * rϕ
    vx = v * cos(θ)
    vy = v * sin(θ) * cos(ϕ)
    vz = v * sin(θ) * sin(ϕ)
    x = rx * vx * Δt
    # assign (x, vx, vy, vz) values to particles matrix
    particles[1, j] = x
    particles[2, j] = vx
    particles[3, j] = vy
    particles[4, j] = vz
end

function run()
  # Create a CUDA array with the random numbers in of size 4xN
  particles = CuArray(rand(Float64, 4, 2^22))
  backend = CUDABackend()
  # make the kernel for this backend
  kernel = knudsencosine!(backend)
  dt = 0.5 # the timestep
  vth = 3.0 # the thermal speed
  # evaluate the kernel
  kernel(particles, dt, vth, ndrange=size(particles, 2))
  # sync
  KernelAbstractions.synchronize(backend)
  # back to host
  hostparticles = Matrix(particles)
  # plot and save
  table = (;zip((:x, :vx, :vy, :vz), eachrow(hostparticles))...)
  h = pairplot(
        table => (PairPlots.Hist(colormap=:magma),
                  PairPlots.MarginDensity()),
       )
  save("knudsencosine.png", h)
end

run()
