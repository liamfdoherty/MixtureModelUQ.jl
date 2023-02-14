using Distributions, Plots

λ1 = 1.; λ2 = 2.
nominal_dist = Exponential(λ1); alternative_dist = Exponential(λ2)

nominal_samples = rand(nominal_dist, 100_000)
alternative_samples = rand(alternative_dist, 100_000)

error = mean(alternative_samples) - mean(nominal_samples)

relative_entropy = log(λ1/λ2) + λ2/λ1 - 1

centered_nominal = nominal_samples .- mean(nominal_samples)

UQII(c; centered_samples) = exp(relative_entropy)^(1/c) * (mean(exp.(c .* centered_samples)))^(1/c)

c_vals = LinRange(1_000, 10_000, 1_000)
plot(c_vals, [exp(error) for c in c_vals], label = "True Error = $(exp(error))",
                                     title = "True Error vs. UQII\nP = Exp($(λ1)), Q = Exp($(λ2))", xlabel = "c")
plot!(c_vals, UQII.(c_vals, centered_samples = centered_nominal), label = "UQII")