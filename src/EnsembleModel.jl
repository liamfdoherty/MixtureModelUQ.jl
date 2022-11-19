mutable struct EnsembleModel{TM}
    model_type::TM
    num_models::Int
    models::Vector
    machines::Vector
end

function EnsembleModel(model_type::DataType, num_models::Int)
    models = [model_type() for _ in 1:num_models]
    return EnsembleModel(model_type, num_models, models, [])
end

function construct_machines!(ensemble::EnsembleModel, mixture::GMM, data_size::Int)
    datasets = []
    for _ = 1:ensemble.num_models
        dataset = DataFrame(samples = Float64[], components = Float64[])
        for __ = 1:data_size
            sample, component = select_and_sample(mixture)
            push!(dataset, [sample, component])
        end
        push!(datasets, dataset)
    end
    for (i, model) in enumerate(ensemble.models)
        dataset = datasets[i]
        X = dataset[!, [:samples]]
        y = coerce(dataset.components, Multiclass)
        mach = machine(model, X, y)
        push!(ensemble.machines, mach)
    end
end

function fit!(ensemble::EnsembleModel)
    for (i, mach) in enumerate(ensemble.machines)
        ensemble.machines[i] = MLJ.fit!(mach)
    end
end

function predict(ensemble::EnsembleModel, samples; mode = false)
    predictions = []
    for mach in ensemble.machines
        if mode == false
            prediction = MLJ.predict(mach, DataFrame(s = samples))
        else
            prediction = MLJ.predict_mode(mach, DataFrame(s = samples))
        end
        push!(predictions, prediction)
    end
    return predictions
end