using Distributions, Plots

λ1 = 500.; λ2 = 100_000.
nominal_dist = Exponential(λ1); alternative_dist = Exponential(λ2)

nominal_samples = rand(nominal_dist, 100_000)
alternative_samples = rand(alternative_dist, 100_000)

error = mean(alternative_samples) - mean(nominal_samples)

relative_entropy = log(λ1/λ2) + λ2/λ1 - 1

centered_nominal = nominal_samples .- mean(nominal_samples)

cgf(c, centered_samples) = c*maximum(centered_samples) + log(mean(exp.(c .* (centered_samples .- maximum(centered_samples)))))
UQII(c) = (1/c) * (cgf(c, centered_nominal) + relative_entropy)

c_vals = LinRange(1, 10_000, 1_000)
plot(c_vals, [error for c in c_vals], label = "True Error",
                                     title = "True Error vs. UQII\nP = Exp($(λ1)), Q = Exp($(λ2))", xlabel = "c")
plot(c_vals, UQII.(c_vals), label = "UQII")


