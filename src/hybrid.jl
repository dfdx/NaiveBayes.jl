"""
    fit(m::HybridNB, f_c::Vector{Vector{Float64}}, f_d::Vector{Vector{Int64}}, labels::Vector{Int64})

Train NB model with discrete and continuous features by estimating P(x⃗|c)
"""
function fit{C, T<: AbstractFloat, U <: Integer, N}(model::HybridNB, continuous_features::Dict{N, Vector{T}}, discrete_features::Dict{N, Vector{U}}, labels::Vector{C})
    A = 1.0/float(length(labels))
    for class in model.classes
        inds = find(labels .== class)
        model.priors[class] = A*float(length(inds))
        for (name, feature) in continuous_features
            f_data = feature[inds]
            model.c_kdes[class][name] = InterpKDE(kde(f_data[isfinite(f_data)]), eps(Float64), InterpLinear)
        end
        for (name, feature) in discrete_features
            f_data = feature[inds]
            model.c_discrete[class][name] = ePDF(f_data[isfinite(f_data)])
        end
    end
    return model
end

"""
    train(HybridNB, continuous, discrete, labels) -> model2
"""
function train{C, T<: AbstractFloat, U <: Integer, N}(::Type{HybridNB}, continuous_features::Dict{N, Vector{T}}, discrete_features::Dict{N, Vector{U}}, labels::Vector{C})
    return fit(HybridNB(labels, N), continuous_features, discrete_features, labels)
end


"""
    fit(m::HybridNB, f_c::Matrix{Float64}, labels::Vector{Int64})

Train NB model with continuous features only
"""
function fit{C, T<: AbstractFloat}(model::HybridNB, continuous_features::Matrix{T}, labels::Vector{C})
    discrete_features = Dict{Symbol, Vector{Int64}}()
    return fit(model, restructure_matrix(continuous_features), discrete_features, labels)
end


"""computes log[P(x⃗ⁿ|c)] ≈ ∑ᵢ log[p(xⁿᵢ|c)] """
function sum_log_x_given_c!{T <: AbstractFloat, U <: Integer, N}(class_prob::Vector{Float64}, feature_prob::Vector{Float64}, m::HybridNB, continuous_features::Dict{N, Vector{T}}, discrete_features::Dict{N, Vector{U}}, c)
    f_prob = zeros(num_kdes(m) + num_discrete(m), num_samples(m, continuous_features, discrete_features))
    for i = 1:num_samples(m, continuous_features, discrete_features)
        for (j, name) in enumerate(keys(continuous_features))
            x_i = continuous_features[name][i]
            if isnan(x_i)
                feature_prob[j] = NaN
            else
                feature_prob[j] = pdf(m.c_kdes[c][name], x_i)
            end
        end

        for (j, name) in enumerate(keys(discrete_features))
            x_i = discrete_features[name][i]
            if isnan(x_i)
                feature_prob[num_kdes(m)+j] = NaN
            else
                feature_prob[num_kdes(m)+j] = probability(m.c_discrete[c][name], x_i)
            end
        end
        f_prob[:, i] = feature_prob
        sel = isfinite(feature_prob)
        class_prob[i] = sum(log(feature_prob[sel]))
    end
    return f_prob
end


""" compute the number of samples """
function num_samples{T <: AbstractFloat, U <: Integer, N}(m::HybridNB, continuous_features::Dict{N, Vector{T}}, discrete_features::Dict{N, Vector{U}}) # TODO: this is a bit strange
    if length(keys(continuous_features)) > 0
        return length(continuous_features[collect(keys(continuous_features))[1]])
    elseif length(keys(discrete_features)) > 0
        return length(discrete_features[collect(keys(discrete_features))[1]])
    else
        return 0
    end
end


"""
    predict_logprobs(m::HybridNB, features_c::Vector{Vector{Float64}, features_d::Vector{Vector{Int})

Return the log-probabilities for each column of X, where each row is the class
"""
function predict_logprobs{T <: AbstractFloat, U <: Integer, N}(m::HybridNB, continuous_features::Dict{N, Vector{T}}, discrete_features::Dict{N, Vector{U}})
    n_samples = num_samples(m, continuous_features, discrete_features)
    log_probs_per_class = zeros(length(m.classes) ,n_samples)
    feature_prob = Vector{Float64}(num_kdes(m) + num_discrete(m))
    feature_probilities = []
    for (i, c) in enumerate(m.classes)
        class_prob = Vector{Float64}(n_samples)
        p = sum_log_x_given_c!(class_prob, feature_prob, m, continuous_features, discrete_features, c)
        log_probs_per_class[i, :] = class_prob .+ log(m.priors[c])
        push!(feature_probilities, p)
    end
    return log_probs_per_class, feature_probilities
end


"""
    predict_proba{V<:Number}(m::HybridNB, f_c::Vector{Vector{Float64}}, f_d::Vector{Vector{Int64}})

Predict log-probabilities for the input features.
Returns tuples of predicted class and its log-probability estimate.
"""
function predict_proba{T <: AbstractFloat, U <: Integer, N}(m::HybridNB, continuous_features::Dict{N, Vector{T}}, discrete_features::Dict{N, Vector{U}})
    logprobs, feature_probilities = predict_logprobs(m, continuous_features, discrete_features)
    n_samples = num_samples(m, continuous_features, discrete_features)
    predictions = Array(Tuple{eltype(m.classes), Float64}, n_samples)
    for i = 1:n_samples
        maxprob_idx = indmax(logprobs[:, i])
        c = m.classes[maxprob_idx]
        logprob = logprobs[maxprob_idx, i]
        predictions[i] = (c, logprob)
    end
    return predictions
end

""" Predict kde naive bayes for continuos featuers only""" # TODO: remove this
function predict{T <: Number}(m::HybridNB, X::Matrix{T})
    eltype(X) <: AbstractFloat || throw("Continuous features must be floats!")
    return predict(m, restructure_matrix(X), Dict{Symbol, Vector{Int}}())
end

"""
    predict(m::HybridNB, f_c::Vector{Vector{Float64}}, f_d::Vector{Vector{Int64}}) -> labels

Predict hybrid naive bayes for continuos featuers only
"""
function predict{T <: AbstractFloat, U <: Integer, N}(m::HybridNB, continuous_features::Dict{N, Vector{T}}, discrete_features::Dict{N, Vector{U}})
    return [k for (k,v) in predict_proba(m, continuous_features, discrete_features)]
end

# TODO remove this once KernelDensity.jl pull request #27 is merged/tagged.
function KernelDensity.InterpKDE{IT<:Grid.InterpType}(k::UnivariateKDE, bc::Number, it::Type{IT}=InterpQuadratic)
    g = CoordInterpGrid(k.x, k.density, bc, it)
    InterpKDE(k, g)
end


function write_model{S <: AbstractString}(model::HybridNB, filename::S)
    h5open(filename, "w") do f
        name_type = eltype(keys(model.c_kdes[model.classes[1]]))
        f["NameType"] = "$name_type"
        info("Writing a model with names of type $name_type")
        grp = g_create(f, "Classes")
        grp["Label"] = model.classes
        grp["Prior"] = collect(values(model.priors))
        for c in model.classes
            grp = g_create(f, "$c")
            sub = g_create(grp, "Discrete")
            for (name, discrete) in model.c_discrete[c]
                f_grp = g_create(sub, "$name")
                f_grp["range"] = collect(keys(discrete.pairs))
                f_grp["probability"] = collect(values(discrete.pairs))
            end
            sub = g_create(grp, "Continuous")
            for (name, continuous) in model.c_kdes[c]
                f_grp = g_create(sub, "$name")
                f_grp["x"] = collect(continuous.kde.x)
                f_grp["density"] = collect(continuous.kde.density)
            end
        end
    end
    info("Writing HybridNB model to file $filename")
end


function to_range{T <: Number}(y::Vector{T})
    min, max = extrema(y)
    dy = (max-min)/(length(y)-1)
    return min:dy:max
end


function load_model{S <: AbstractString}(filename::S)
    model = h5open(filename, "r") do f
        N = read(f["NameType"]) == "Symbol" ? Symbol : AbstractString
        classes = read(f["Classes/Label"])
        C = eltype(classes)
        priors = Dict{C, Float64}()
        [priors[k] =v for (k,v) in zip(classes, read(f["Classes/Prior"]))]

        kdes = Dict{C, Dict{N, InterpKDE}}()
        discrete = Dict{C, Dict{N, ePDF}}()
        for c in classes
            kdes[c] = Dict{N, InterpKDE}()
            for (name, dist) in read(f["$c"]["Continuous"])
                kdes[c][N(name)] = InterpKDE(UnivariateKDE(to_range(dist["x"]), dist["density"]), eps(Float64), InterpLinear)
            end
            discrete[c] = Dict{N, ePDF}()
            for (name, dist) in read(f["$c"]["Discrete"])
                rng = dist["range"]
                prob = dist["probability"]
                d = Dict{eltype(rng), eltype(prob)}()
                [d[k]=v for (k,v) in zip(rng, prob)]
                discrete[c][N(name)] = ePDF(d)
            end
        end
        return HybridNB{C, N}(kdes, discrete, classes, priors)
    end
    return model
end
