module MixtureModelUQ

using StatsBase
using Distributions

include("GMM.jl")
export GMM, select_and_sample

end
