using Distributions, Plots

# Set up the observable and point of evaluation
f(x, coef, intercept) = inv(1 + exp(intercept + coef*x))
p = 0.

# Set up the distributions
nominal_coef = Normal(-5.24, 1.44); nominal_intercept = Normal(0.87, 0.25)
alternative_coef = Normal(-5.21, 1.42); alternative_intercept = Normal(-0.87, 0.25)

# Compute the relative entropy
function relative_entropy(x::Normal, y::Normal)
    m1 = x.μ; m2 = y.μ
    s1 = x.σ; s2 = y.σ
    return log(s2/s1)/2 + (s1^2 + (m1 - m2)^2)/(2*s2^2) - 1/2
end
intercept_rel_entropy = relative_entropy(alternative_intercept, nominal_intercept)
coef_rel_entropy = relative_entropy(alternative_coef, nominal_coef)
rel_entropy = intercept_rel_entropy + coef_rel_entropy

# Monte Carlo estimate of true error
N = 10_000
nominal_samples = collect(zip(rand(nominal_coef, N), rand(nominal_intercept, N)))
alternative_samples = collect(zip(rand(alternative_coef, N), rand(alternative_intercept, N)))

nominal_observables = [f(p, c, i) for (c, i) in nominal_samples]
alternative_observables = [f(p, c, i) for (c, i) in alternative_samples]

error = mean(alternative_observables) - mean(nominal_observables)

# Get the centered observable and stable CGF function
centered_observables = nominal_observables .- mean(nominal_observables)

stable_cgf(c) = c*maximum(centered_observables) + log(mean(exp.(c .* (centered_observables .- maximum(centered_observables)))))

# Get the correction (bias term from logarithm)
risk_sensitive_samples(c) = exp.(c .* (centered_observables .- maximum(centered_observables)))
bias(c) = var(risk_sensitive_samples(c))/(2*N*mean(risk_sensitive_samples(c))^2)

# Get the Θ with bias correction
Θ(c) = (1/c)*(stable_cgf(c) + bias(c) + rel_entropy)

# Plot
c_vals = LinRange(10, 1_000_000, 100)
plot(c_vals, Θ.(c_vals), label = "Θ", title = "Monte Carlo Estimate, N = $(N)", ylims = (0, 2.5))
plot!(c_vals, [error for c in c_vals], label = "True Error")