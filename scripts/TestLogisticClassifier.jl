using Distributions, QuadGK
include("GenerateData.jl")
println("Loaded Code.")

N_vals = [100]
LC = @load LogisticClassifier pkg = MLJLinearModels
synthetic_data = LinRange(-1, 1, 101)

# Collect the positive probabilities for each N in matrices (row = data point, column = model)
ensembles, positive_probs = generate_data(N_vals, LC, synthetic_data, num_models = 10_000)
probs = reduce(hcat, positive_probs[N_vals[1]][1])

# Collect the fitted parameters and fit them
coef_dict = Dict(); intercept_dict = Dict()
for N_val in N_vals
    nominal_machs = ensembles[N_val][1].machines
    alternative_machs = ensembles[N_val][2].machines
    machs = Dict("Nominal" => nominal_machs, "Alternative" => alternative_machs)
    coef_dict_N = Dict(); intercept_dict_N = Dict()
    for (collection_label, mach_collection) in machs
        coefs = Vector{Float64}(); intercepts = Vector{Float64}()
        for mach in mach_collection
            push!(coefs, fitted_params(mach).coefs[1][2])
            push!(intercepts, fitted_params(mach).intercept)
        end

        # Fit a distribution to the fitted parameters (they appear normal) and plot
        coef_distribution = fit(Normal, coefs)
        intercept_distribution = fit(Normal, intercepts)

        coef_plot = histogram(coefs, title = "$(collection_label) Model Coefficients", label = "N = $(N_val)",
                                 legend = :outertopright, bins = 50, normalize = :pdf)
        coef_x = LinRange(minimum(coefs), maximum(coefs), 101)
        plot!(coef_x, pdf.(coef_distribution, coef_x), primary = false, color = :red)

        intercept_plot = histogram(intercepts, title = "$(collection_label) Model Intercepts", label = "N = $(N_val)",
                                     legend = :outertopright, bins = 50, normalize = :pdf)
        intercept_x = LinRange(minimum(intercepts), maximum(intercepts), 101)
        plot!(intercept_x, pdf.(intercept_distribution, intercept_x), primary = false, color = :red)

        display(coef_plot)
        display(intercept_plot)

        coef_dict_N[collection_label] = Dict("Data" => coefs, "Distribution" => coef_distribution)
        intercept_dict_N[collection_label] = Dict("Data" => intercepts, "Distribution" => intercept_distribution)
    end
    coef_dict[N_val] = coef_dict_N
    intercept_dict[N_val] = intercept_dict_N
end


# Estimate the UQIIs and plot
f(x, coef, intercept) = inv(1 + exp(intercept + coef*x))

function mean_2d_quadrature(f::Function, x_dist::Distribution, y_dist::Distribution) # Helper function for means
    return quadgk(y -> quadgk(x -> f(x, y)*pdf(x_dist, x), -Inf, Inf)[1]*pdf(y_dist, y), -Inf, Inf)[1]
end

for N_val in N_vals
    nominal_coefs = coef_dict[N_val]["Nominal"]["Data"]
    nominal_intercepts = intercept_dict[N_val]["Nominal"]["Data"]
    nominal_parameters = zip(nominal_coefs, nominal_intercepts)

    alternative_coefs = coef_dict[N_val]["Alternative"]["Data"]
    alternative_intercepts = intercept_dict[N_val]["Alternative"]["Data"]
    alternative_parameters = zip(alternative_coefs, alternative_intercepts)
    
    error(x) = mean([f(x, c, i) for (c, i) in alternative_parameters]) - mean([f(x, c, i) for (c, i) in nominal_parameters])
    display(plot(synthetic_data, error.(synthetic_data), label = "Error, N = $(N_val)", legend = :outertopright))
end

function relative_entropy(x::Normal, y::Normal)
    m1 = x.μ; m2 = y.μ
    s1 = x.σ; s2 = y.σ
    return log(s2/s1)/2 + (s1^2 + (m1 - m2)^2)/(2*s2^2) - 1/2
end

rel_entropies = Dict()
for N_val in N_vals
    intercept_rel_entropy = relative_entropy(intercept_dict[N_val]["Alternative"]["Distribution"], intercept_dict[N_val]["Nominal"]["Distribution"])
    coef_rel_entropy = relative_entropy(coef_dict[N_val]["Alternative"]["Distribution"], coef_dict[N_val]["Nominal"]["Distribution"])
    rel_entropy = intercept_rel_entropy + coef_rel_entropy
    rel_entropies[N_val] = rel_entropy
end

# Get (nominal) centered observables at p = 0
centered_observables = Dict()
p = 0.
for N_val in N_vals
    coef_and_intercept = collect(zip(coef_dict[N_val]["Nominal"]["Data"], intercept_dict[N_val]["Nominal"]["Data"]))
    observable = [f(p, coef, intercept) for (coef, intercept) in coef_and_intercept]
    centered_observable = observable .- mean(observable)
    centered_observables[N_val] = centered_observable
end

# Get CGF
CGFs = Dict()
for N_val in N_vals
    stable_cgf(c, data) = c*maximum(data) + log(mean(exp.(c .* (data .- maximum(data)))))
    CGFs[N_val] = c -> stable_cgf(c, centered_observables[N_val])
end

# Get Θ functions
Θs = Dict()
for N_val in N_vals
    Θs[N_val] = c -> (1/c)*(CGFs[N_val](c) + rel_entropies[N_val])
end
