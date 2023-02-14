using Distributions, Plots
using Random; Random.seed!(1)

σ = 1.
dist = Normal(0, σ)
# p = 0.5
# dist = Bernoulli(p)
EX = mean(dist)
true_value(c; ρ) = c*σ^2/2 + ρ^2/c

for N in Int.([1e3, 1e4, 1e5, 1e6])
    samples = rand(dist, N)
    samples_offset = maximum(samples);
    approximation(c; ρ) = (samples_offset - EX) + (1/c)*log(mean(exp.(c .* (samples .- (samples_offset + EX))))) + ρ^2/c

    c_vals = LinRange(1, 100, 1000)
    anim = @animate for ρ in 0.1:0.1:10
        # plot(c_vals, [true_value.(c_vals, ρ = ρ), approximation.(c_vals, ρ = ρ)], xlabel = "c", 
        #             label = ["Truth" "Approximation"], legend = :outertopright, title = "N = $(N), ρ = $(ρ)")
        plot(c_vals, approximation.(c_vals, ρ = ρ), xlabel = "c", 
                    label = "Approximation", legend = :outertopright, title = "N = $(N), ρ = $(ρ)")
    end

    gif(anim, "ThetaAnimationScaled.gif", fps = 15)
end