using StatsPlots, Distributions, DataFrames, Plots
using MixtureModelUQ

components = [Normal(-0.1, 0.2), Normal(0.1, 0.2)]

weight1 = [0.7, 0.3]; weight2 = [0.3, 0.7]

model1 = GMM(components, weight1)
model2 = GMM(components, weight2)

# Collect samples for both models
N_samples = 100000

model1_data = DataFrame(samples = Float64[], components = Float64[])
for i = 1:N_samples
    sample, component = select_and_sample(model1)
    push!(model1_data, [sample, component])
end

model2_data = DataFrame(samples = Float64[], components = Float64[])
for i = 1:N_samples
    sample, component = select_and_sample(model2)
    push!(model2_data, [sample, component])
end

# Visualize data
xvals = LinRange(-1, 1, 1001)

plt1 = histogram(model1_data.samples, title = "Model 1", xlabel = "x", normalize = :pdf, label = "Samples")
pdf1(x) = weight1[1]*pdf(components[1], x) + weight1[2]*pdf(components[2], x)
plot!(xvals, pdf1.(xvals), label = "PDF")
display(plt1)

plt2 = histogram(model2_data.samples, title = "Model 2", xlabel = "x", normalize = :pdf, label = "Samples")
pdf2(x) = weight2[1]*pdf(components[1], x) + weight2[2]*pdf(components[2], x)
plot!(xvals, pdf2.(xvals), label = "PDF")
display(plt2)

# Estimate relative entropy
f(x) = weight1[1]*pdf(components[1], x) + weight1[2]*pdf(components[2], x)
g(x) = weight2[1]*pdf(components[1], x) + weight2[2]*pdf(components[2], x)
x = model2_data.samples
rel_entropy = mean(log.(g.(x) ./ f.(x)))