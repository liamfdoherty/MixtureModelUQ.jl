using LogExpFunctions
include("GenerateData.jl")
println("Loaded Code.")

N_vals = [1_000]
LC = @load LogisticClassifier pkg = MLJLinearModels
synthetic_data = LinRange(-1, 1, 101)

# Collect the positive probabilities for each N in matrices (row = data point, column = model)
ensembles, positive_probs = generate_data(N_vals, LC, synthetic_data)
nominal_probs = Dict(label => reduce(hcat, probs[1]) for (label, probs) in positive_probs)
alternative_probs = Dict(label => reduce(hcat, probs[2]) for (label, probs) in positive_probs)

# Center the nominal observables pointwise
centered_nominals = Dict()
for (N, prob_matrix) in nominal_probs
    centered_nominal = similar(prob_matrix)
    for i in axes(prob_matrix, 1)
        @views centered_nominal[i, :] .= prob_matrix[i, :] .- mean(prob_matrix[i, :])
    end
    centered_nominals[N] = centered_nominal
end

# Plot the true UQ error
error_curves = Dict()
for N in keys(positive_probs)
    nominal = nominal_probs[N]
    alternative = alternative_probs[N]
    
    errors = mean.(eachrow(alternative)) - mean.(eachrow(nominal))
    error_curves[N] = errors
end

error_plot = plot(legend = :outertopright)
for (N, errors) in error_curves
   plot!(synthetic_data, errors, label = "Error, N = $(N)")
end

# Compute and plot the linearized bounds
rel_entropy = 0.34 # relative entropy for single sample distributions
for N in keys(positive_probs)
    linearized_bounds = sqrt(2*rel_entropy*N) .* sqrt.(var.(eachrow(nominal_probs[N])))
    plot!(synthetic_data, linearized_bounds, label = "Lin Bounds, N = $(N)")
end

# Compute the upper UQ bounds; something is wrong here
compute_cgf(c; centered_data) = c*maximum(centered_data) + log(mean(exp.(c .* (centered_data .- maximum(centered_data)))))
bias(c; centered_data, N) = var(exp.(c .* (centered_data .- maximum(centered_data))))/(2*N*mean(exp.(c .* (centered_data .- maximum(centered_data))))^2)
c_vals = [100_000]
for N in keys(positive_probs)
    Θ(c; centered_data, N) = (1/c)*(compute_cgf(c, centered_data = centered_data) + bias(c, centered_data = centered_data, N = N) + rel_entropy*N)
    for c in c_vals
        upper_bounds = [Θ(c, centered_data = centered_data, N = N) for centered_data in eachrow(centered_nominals[N])]
        plot!(synthetic_data, upper_bounds, label = "UQII, N = $(N), c = $(c)")
    end
end

display(error_plot)