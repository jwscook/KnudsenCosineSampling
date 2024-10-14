using CUDA, KernelAbstractions, StatsBase, CairoMakie, PairPlots

pade(x::T, numer, denom) where T = T(evalpoly(x, numer) * x / (1 + evalpoly(x, denom) * x))

@kernel function knudsencosine!(particles, Δt)
    numer = (1.1892074873464582, 2.3884028521993717, 0.021742384140688395,
             1.4551991003478761,−0.01545968501315156, 0.25068176822296523)
    denom = (2.0084051736325055, −0.2174611175058831, 0.7504356620416117,
             0.038026260588022796, 0.03376005807862799, −0.002372131900552844)
    j = @index(Global)
    rx = particles[1, j]
    rv = particles[2, j]
    rθ = particles[3, j]
    rϕ = particles[4, j]
    z = sqrt(sqrt(-log(1 - rv)))
    v = pade(z, numer, denom)
    θ = acos(cbrt(1 - rθ))
    ϕ = 2π * rϕ
    vx = v * cos(θ)
    vy = v * sin(θ) * cos(ϕ)
    vz = v * sin(θ) * sin(ϕ)
    x = rx * vx * Δt
    particles[1, j] = x
    particles[2, j] = vx
    particles[3, j] = vy
    particles[4, j] = vz
end

const particles = CuArray(rand(Float64, 4, 2^22))
const backend = CUDABackend()
kernel = knudsencosine!(backend)
kernel(particles, 1.0, ndrange=size(particles, 2))
KernelAbstractions.synchronize(backend)

hostparticles = Matrix(particles)

table = (;zip((:x, :vx, :vy, :vz), eachrow(hostparticles))...)
h = pairplot(
      table => (PairPlots.Hist(colormap=:magma),
                PairPlots.MarginDensity()),
     )
save("knudsencosine.png", h)
