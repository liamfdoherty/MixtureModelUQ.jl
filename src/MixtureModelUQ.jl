module MixtureModelUQ

using StatsBase
using Distributions
using DataFrames
using MLJ

include("GMM.jl")
export GMM, select_and_sample

include("EnsembleModel.jl")
export EnsembleModel, construct_machines!, fit!, predict

end
