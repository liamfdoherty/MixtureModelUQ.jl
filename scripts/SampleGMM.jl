using StatsPlots, Distributions, DataFrames
using MixtureModelUQ

components = [Normal(-0.1, 0.1), Normal(0.1, 0.1)]

weight1 = [0.7, 0.3]; weight2 = [0.3, 0.7]

model1 = GMM(components, weight1)
model2 = GMM(components, weight2)

# Collect samples for both models
model1_data = DataFrame(samples = Float64[], components = Float64[])
N_samples = 1000
for i = 1:N_samples
    sample, component = select_and_sample(model1)
    push!(model1_data, [sample, component])
end

model2_data = DataFrame(samples = Float64[], components = Float64[])
N_samples = 1000
for i = 1:N_samples
    sample, component = select_and_sample(model2)
    push!(model2_data, [sample, component])
end

# Visualize data
display(histogram(model1_data.samples, title = "Model 1"))
display(histogram(model2_data.samples, title = "Model 2"))