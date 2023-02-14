using Distributions, MLJ, Optimization, Plots, LaTeXStrings
using MixtureModelUQ
using Random; Random.seed!(1)

function generate_data(N_vals, model, synthetic_data; num_models = 100)
    probs_dict = Dict()
    ensembles_dict = Dict()
    for num_samples in N_vals
        components = [Normal(-0.1, 0.2), Normal(0.1, 0.2)]

        weight1 = [0.7, 0.3]; weight2 = [0.3, 0.7]
        mixture1 = GMM(components, weight1); mixture2 = GMM(components, weight2)
        mixtures = [mixture1, mixture2]

        ensemble1 = MixtureModelUQ.EnsembleModel(model, num_models)
        ensemble2 = MixtureModelUQ.EnsembleModel(model, num_models)
        ensembles = [ensemble1, ensemble2]

        positive_probs = []

        for (ensemble, mixture) in zip(ensembles, mixtures)
            construct_machines!(ensemble, mixture, num_samples)

            MixtureModelUQ.fit!(ensemble)

            predictions = MixtureModelUQ.predict(ensemble, synthetic_data, mode = false)

            positive_probabilities = [[prediction.prob_given_ref[1] for prediction in model_predictions] for model_predictions in predictions]
            push!(positive_probs, positive_probabilities)
        end
        probs_dict[num_samples] = positive_probs
        ensembles_dict[num_samples] = ensembles
    end
    return ensembles_dict, probs_dict
end
