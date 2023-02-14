using Distributions, Plots, LaTeXStrings
using Random; Random.seed!(1)

σ_vals = LinRange(1, 10, 500)
mean_errors = []
cgf_errors = []
biases = []
N = Int(1e5)
c = 1.
for σ in σ_vals
    # Set up the problem with the distribution and the observable
    dist = Normal(0, σ)
    F(x) = exp(c*x) # The observable is really just x, and we are fixing c = 1

    # The sample mean is the approximation of E[e^(cx)]
    samples = rand(dist, N)
    sample_mean = mean(F.(samples))

    # This is the analytical mean
    true_mean = exp((c*σ)^2/2)

    # The relative error in the means
    mean_error = abs(true_mean - sample_mean)/abs(true_mean)

    # Now we compute the cgfs and their relative error
    true_cgf = log(true_mean)
    sample_cgf = log(sample_mean)
    cgf_error = abs(true_cgf - sample_cgf)/abs(true_cgf)

    # The bias introduced when taking the logarithm
    bias = (exp((c*σ)^2) - 1)/(2*N)

    push!(mean_errors, mean_error)
    push!(cgf_errors, cgf_error)
    push!(biases, bias)
end

mean_plot = plot(σ_vals, mean_errors,
                 title = L"Relative Error in $\mathbb{E}_{N(0, \sigma^2)}[e^X]$", 
                 xlabel = L"\sigma", ylabel = L"(\bar{F}_{N} - \mathbb{E}[F])/\mathbb{E}[F]",
                 label = "Error", legend = :outertopright, yscale = :log10)

cgf_plot = plot(σ_vals, cgf_errors,
                title = L"Relative Error in $\log\mathbb{E}_{N(0, \sigma^2)}[e^X]$", 
                xlabel = L"\sigma", ylabel = L"(\log\bar{F}_{N} - \log\mathbb{E}[F])/\log\mathbb{E}[F]",
                label = "Error", legend = :outertopright)

bias_plot = plot(σ_vals, log.(biases),
                title = L" Log of Bias in $\log\mathbb{E}_{N(0, \sigma^2)}[e^X]$", 
                xlabel = L"\sigma", ylabel = L"(\log\bar{F}_{N} - \log\mathbb{E}[F])/\log\mathbb{E}[F]",
                label = "Bias", legend = :outertopright)

display(mean_plot)
display(cgf_plot)
display(bias_plot)