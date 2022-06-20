using Copulas, Distributions, Random, Turing, Plots, StatsPlots

Random.seed!(123)
D = SklarDist(ClaytonCopula(3,7), (Exponential(1.0), Pareto(3.0), Exponential(2.0)))
draws = rand(D, 2_000)

@model function copula(X)
    # Priors on rate parameters
    θ  ~ TruncatedNormal(1.0, 1.0, 0, Inf)
    θ₁ ~ TruncatedNormal(1.0, 1.0, 0, Inf)
    θ₂ ~ TruncatedNormal(1.0, 1.0, 0, Inf)
    θ₃ ~ TruncatedNormal(1.0, 1.0, 0, Inf)

    # Marginal distributions and copula
    X₁ = Exponential(θ₁)
    X₂ = Pareto(θ₂)
    X₃ = Exponential(θ₃)
    C = ClaytonCopula(3,θ)
    D = SklarDist(C, (X₁, X₂, X₃))
    Turing.Turing.@addlogprob! loglikelihood(D, X)
end

sampler = NUTS() # MH() works too
chain = sample(copula(draws), sampler, MCMCThreads(), 1_00, 4)
p = plot(chain)
savefig(p, "plot_archimedeans.png")