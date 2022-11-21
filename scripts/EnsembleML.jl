using Distributions, MLJ
using MixtureModelUQ
using Random; Random.seed!(1)

components = [Normal(-0.1, 0.2), Normal(0.1, 0.2)]

weight = [0.7, 0.3]
mixture = GMM(components, weight)

LC = @load LogisticClassifier pkg = MLJLinearModels

ensemble = MixtureModelUQ.EnsembleModel(LC, 50)

construct_machines!(ensemble, mixture, 1000)

MixtureModelUQ.fit!(ensemble)

synthetic_data = LinRange(-1, 1, 101)
mode_choice = false
predictions = MixtureModelUQ.predict(ensemble, synthetic_data, mode = mode_choice)

if mode_choice == false
    positive_probabilities = [[prediction.prob_given_ref[1] for prediction in model_predictions] for model_predictions in predictions]
    
    display(plot(synthetic_data, positive_probabilities, title = "Probability of Positive Class for IID Models",
                xlabel = "x", ylabel = "P(k = 1)", legend = :none))

    display(plot(synthetic_data, mean(positive_probabilities), ribbon = std(positive_probabilities),
                title = "Mean Confidence with One Standard Deviation", xlabel = "x", ylabel = "P(k = 1)", legend = :none))
end
