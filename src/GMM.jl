struct GMM
    num_components::Int            # Number of components
    d::Int                         # dimension of Gaussian
    w::Vector{Float64}             # weights: num_components
    components::Vector             # components of mixture
end

function GMM(components::Vector, w::Vector)
    @assert eltype(components) <: Normal || eltype(components) <: MvNormal "Components must be Gaussian!"
    @assert length(w) == length(components) "Number of components and weights must be equal!"
    @assert sum(w) == 1 && all(0. .<= w .<= 1.) "Weight vector must be a probability vector!"
    @assert length(Set(length.(components))) <= 1  "Dimension of Gaussian components must all be equal!"
    
    num_components = length(components)
    d = length(components[1])
    return GMM(num_components, d, w, components)
end

function select_and_sample(model::GMM)
    d = Bernoulli(model.w[1]) # probability of choosing first component
    choice = rand(d)
    component = choice == true ? model.components[1] : model.components[2]
    s = rand(component)
    return s, choice
end