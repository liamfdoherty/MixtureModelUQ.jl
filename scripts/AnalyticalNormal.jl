using Distributions, Plots, Optim
using LaTeXStrings
using LinearAlgebra
using Random

"""
`compute_error` - computes the error in the MC calculated minimizer of the UQII vs the truth; returns truth, approximate and error
"""
function compute_error(σ, ρ, N, upper_bound)
    Random.seed!(1)
    # Gether samples from the distribution
    dist = Normal(0, σ)
    samples = rand(dist, N)

    # Function for computing the CGF and UQII
    sampled_cgf(c) = c*maximum(samples) + log(mean(exp.(c .* (samples .- maximum(samples)))))
    sampled_Θ(ρ, c) = (1/c)*(sampled_cgf(c) + ρ^2)
    sampled_Ξ(ρ) = optimize(c -> sampled_Θ(ρ, c), 0, upper_bound)

    # Functions for the exact values
    exactUQII(ρ) = √(2*σ^2*ρ^2)
    exact_minimizer(σ, ρ) = √(2ρ^2/σ^2)

    # Compute the true minimizer and the approximated minimizer
    true_minimizer = exact_minimizer(σ, ρ)
    approximate_minimizer = sampled_Ξ(ρ).minimizer
    error = approximate_minimizer - true_minimizer
    
    return true_minimizer, approximate_minimizer, error
end

σ_vals = collect(1:15); ρ_vals = collect(1:0.1:5)
N = Int(1e5); upper_bound = 500
true_vals = []; approximate_vals = []; errors = []
σ_ρ_pairs = []
for σ in σ_vals
    for ρ in ρ_vals
        true_minimizer, approximate_minimizer, error = compute_error(σ, ρ, N, upper_bound)
        push!(true_vals, true_minimizer); push!(approximate_vals, approximate_minimizer); push!(errors, error)
        push!(σ_ρ_pairs, (σ, ρ))
    end
end

true_matrix = reshape(true_vals, (length(ρ_vals), length(σ_vals)))
errors_matrix = reshape(errors, (length(ρ_vals), length(σ_vals)))
σ_ρ_matrix = reshape(σ_ρ_pairs, (length(ρ_vals), length(σ_vals)))
display(contour(σ_vals, ρ_vals, true_matrix, fill = true,
        title = L"True $c_{*}$", xlabel = L"\sigma", ylabel = L"\rho"))
display(contour(σ_vals, ρ_vals, errors_matrix, fill = true,
        title = L"Error in $c_{*}$", xlabel = L"\sigma", ylabel = L"\rho"))