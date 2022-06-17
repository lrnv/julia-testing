using Copulas, Distributions, Random, Turing, Plots, StatsPlots

Random.seed!(123)
M₁ = Exponential(1.0)
M₂ = Exponential(1.0)
ρ = 0.5
Σ = [1.0 ρ; ρ 1.0]
C = GaussianCopula(Σ)
D = SklarDist(C, (M₁, M₂))
draws = rand(D, 2_000)

@model function copula(X)
    # Priors on rate parameters for each
    # marginal exponential distribution
    θ₁ ~ TruncatedNormal(1.0, 1.0, 0, Inf)
    θ₂ ~ TruncatedNormal(1.0, 1.0, 0, Inf)

    # Flat prior on Gaussian Copula correlation parameter
    ρ ~ Uniform(-0.95, 0.95)

    # Marginal distributions and 
    # Gaussian copula with correlation parameter ρ
    M₁ = Exponential(θ₁)
    M₂ = Exponential(θ₂)
    C = GaussianCopula([1.0 ρ; ρ 1.0])
    D = SklarDist(C, (M₁, M₂))

    Turing.Turing.@addlogprob! loglikelihood(D, X)
end

sampler = NUTS() # MH() works also.
chain = sample(copula(draws), sampler, MCMCThreads(), 1_00, 4)
p = plot(chain)
savefig(p, "plot.png")