using Distributions, Integrals, Optimization, OptimizationOptimJL, Plots

# Set up the observable and point of evaluation (p is point of evaluation, x = [intercept, coef])
f(x, p) = inv(1 + exp(x[1] + x[2]*p[1]))
p = [0.]

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

# True error computed with quadrature
nominal_integrand(x, p) = f(x, p)*pdf(nominal_intercept, x[1])*pdf(nominal_coef, x[2])
nominal_prob = IntegralProblem(nominal_integrand, [-Inf, -Inf], [Inf, Inf], p)
𝔼μf = solve(nominal_prob, HCubatureJL(), abstol=1e-9).u

alternative_integrand(x, p) = f(x, p)*pdf(alternative_intercept, x[1])*pdf(alternative_coef, x[2])
alternative_prob = IntegralProblem(alternative_integrand, [-Inf, -Inf], [Inf, Inf], p)
𝔼νf = solve(alternative_prob, HCubatureJL(), abstol=1e-9).u

error = 𝔼νf - 𝔼μf

# Get the stable CGF function (p = [x, c])
centered_f(x, p) = inv(1 + exp(x[1] + x[2]*p[1])) - 𝔼μf
f_to_opt(x, p) = -centered_f(x, p)
u0 = [1., 1.]
optim_prob = OptimizationProblem(f_to_opt, u0, p)
max_val = -solve(optim_prob, NelderMead()).minimum
Θ_vals = []
c_vals = LinRange(10, 1000, 1000)
println("Integrating...")
for c in c_vals
    cgf_p = [p[1], c]
    # max_val = 0.
    cgf_integrand(x, p) = exp(p[2]*(centered_f(x, p) - max_val))*pdf(nominal_intercept, x[1])*pdf(nominal_coef, x[2])
    cgf_prob = IntegralProblem(cgf_integrand, [-Inf, -Inf], [Inf, Inf], cgf_p)
    cgf_sol = solve(cgf_prob, HCubatureJL(), abstol=1e-9).u
    cgf_estimate = max_val*c + log(cgf_sol)
    
    # Get the Θ
    Θ = (1/c)*(cgf_estimate + rel_entropy)
    println("Θ($(c)) = $(Θ)")
    push!(Θ_vals, Θ)
end

# Plot
plot(c_vals, Θ_vals, label = "Θ", title = "Quadrature Estimate, x = $(p[1])", xlabel = "c", ylims = (-0.1, 2.5))
plot!(c_vals, [error for c in c_vals], label = "True Error")