using Distributions, MLJ
using MixtureModelUQ

components = [Normal(-0.1, 0.1), Normal(0.1, 0.1)]

weight = [0.7, 0.3]
mixture = GMM(components, weight)

LC = @load LogisticClassifier pkg = MLJLinearModels

ensemble = MixtureModelUQ.EnsembleModel(LC, 10)

construct_machines!(ensemble, mixture, 1000)

MixtureModelUQ.fit!(ensemble)

synthetic_data = LinRange(-1, 1, 101)
predictions = MixtureModelUQ.predict(ensemble, synthetic_data, mode = true)
