"""
    fit(m::HybridNB, f_c::Vector{Vector{Float64}}, f_d::Vector{Vector{Int64}}, labels::Vector{Int64})

Train NB model with discrete and continuous features by estimating P(x⃗|c)
"""
function fit{C, T<: AbstractFloat, U <: Integer}(model::HybridNB, continuous_features::Vector{Vector{T}}, discrete_features::Vector{Vector{U}}, labels::Vector{C})
    for class in model.classes
        inds = find(labels .== class)
        for (j, feature) in enumerate(continuous_features)
            f_data = feature[inds]
            model.c_kdes[class][j] = InterpKDE(kde(f_data[isfinite(f_data)]), eps(Float64), InterpLinear)
        end
        for (j, feature) in enumerate(discrete_features)
            f_data = feature[inds]
            model.c_discrete[class][j] = ePDF(f_data[isfinite(f_data)])
        end
    end
    return model
end


"""
    fit(m::HybridNB, f_c::Matrix{Float64}, labels::Vector{Int64})

Train NB model with continuous features only
"""
function fit{C, T<: AbstractFloat}(model::HybridNB, continuous_features::Matrix{T}, labels::Vector{C})
    discrete_features = Vector{Vector{Int64}}()
    return fit(model, restructure_matrix(continuous_features), discrete_features, labels)
end


"""computes log[P(x⃗ⁿ|c)] ≈ ∑ᵢ log[p(xⁿᵢ|c)] """
function sum_log_x_given_c!{T <: AbstractFloat, U <: Integer}(class_prob::Vector{Float64}, feature_prob::Vector{Float64}, m::HybridNB, continuous_features::Vector{Vector{T}}, discrete_features::Vector{Vector{U}}, c)
    for i = 1:num_samples(m, continuous_features, discrete_features)

        for j = 1:num_kdes(m)
            if isnan(continuous_features[j][i])
                feature_prob[j] = NaN
            else
                feature_prob[j] = pdf(m.c_kdes[c][j], continuous_features[j][i])
            end
        end

        for j = 1:num_discrete(m)
            if isnan(continuous_features[j][i])
                feature_prob[num_kdes(m)+j] = NaN
            else
                feature_prob[num_kdes(m)+j] = probability(m.c_discrete[c][j], discrete_features[j][i])
            end
        end

        sel = isfinite(feature_prob)
        class_prob[i] = sum(log(feature_prob[sel]))
    end
end


""" compute the number of samples """
function num_samples{T <: AbstractFloat, U <: Integer}(m::HybridNB, continuous_features::Vector{Vector{T}}, discrete_features::Vector{Vector{U}}) # TODO: this is a bit strange
    if num_kdes(m) > num_discrete(m)
        n_samples = length(continuous_features[1])
    else
        n_samples = length(discrete_features[1])
    end
    return n_samples
end


"""
    predict_logprobs(m::HybridNB, features_c::Vector{Vector{Float64}, features_d::Vector{Vector{Int})

Return the log-probabilities for each column of X, where each row is the class
"""
function predict_logprobs{T <: AbstractFloat, U <: Integer}(m::HybridNB, continuous_features::Vector{Vector{T}}, discrete_features::Vector{Vector{U}})
    n_samples = num_samples(m, continuous_features, discrete_features)
    log_probs_per_class = zeros(length(m.classes) ,n_samples)
    feature_prob = Vector{Float64}(num_kdes(m) + num_discrete(m))
    for (i, c) in enumerate(m.classes)
        class_prob = Vector{Float64}(n_samples)
        sum_log_x_given_c!(class_prob, feature_prob, m, continuous_features, discrete_features, c)
        log_probs_per_class[i, :] = class_prob .+ log(m.priors[c])
    end
    return log_probs_per_class
end


"""
    predict_proba{V<:Number}(m::HybridNB, f_c::Vector{Vector{Float64}}, f_d::Vector{Vector{Int64}})

Predict log-probabilities for the input features.
Returns tuples of predicted class and its log-probability estimate.
"""
function predict_proba{T <: AbstractFloat, U <: Integer}(m::HybridNB, continuous_features::Vector{Vector{T}}, discrete_features::Vector{Vector{U}})
    logprobs = predict_logprobs(m, continuous_features, discrete_features)
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

""" Predict kde naive bayes for continuos featuers only"""
function predict{T <: Number}(m::HybridNB, X::Matrix{T})
    eltype(X) <: AbstractFloat || throw("Continuous features must be floats!")
    return predict(m, restructure_matrix(X), Vector{Vector{Int}}())
end

"""
    predict(m::HybridNB, f_c::Vector{Vector{Float64}}, f_d::Vector{Vector{Int64}}) -> labels

Predict hybrid naive bayes for continuos featuers only
"""
function predict{T <: AbstractFloat, U <: Integer}(m::HybridNB, continuous_features::Vector{Vector{T}}, discrete_features::Vector{Vector{U}})
    return [k for (k,v) in predict_proba(m, continuous_features, discrete_features)]
end

# TODO remove this once KernelDensity.jl pull request #27 is merged/tagged.
function KernelDensity.InterpKDE{IT<:Grid.InterpType}(k::UnivariateKDE, bc::Number, it::Type{IT}=InterpQuadratic)
    g = CoordInterpGrid(k.x, k.density, bc, it)
    InterpKDE(k, g)
end


function write_model(model::HybridNB, filename::AbstractString)
    h5open(filename, "w") do f
        f["NameType"] = "$(eltype(model.kde_names))"
        info("Writing a model with names of type $(eltype(model.kde_names))")
        grp = g_create(f, "Classes")
        grp["Label"] = model.classes
        grp["Prior"] = collect(values(model.priors))
        for c in model.classes
            grp = g_create(f, "$c")
            sub = g_create(grp, "Discrete")
            for (name, discrete) in zip(model.discrete_names, model.c_discrete[c])
                f_grp = g_create(sub, "$name")
                f_grp["range"] = collect(keys(discrete.pairs))
                f_grp["probability"] = collect(values(discrete.pairs))
            end

            sub = g_create(grp, "Continuous")
            for (name, continuous) in zip(model.kde_names, model.c_kdes[c])
                f_grp = g_create(sub, "$name")
                f_grp["x"] = collect(continuous.kde.x)
                f_grp["density"] = collect(continuous.kde.density)
            end
        end
    end
    info("Writing HybridNB model to file $filename")
end


function get_feature_names(filename::AbstractString)
    classes, priors, kde_names, discrete_names = h5open(filename, "r") do f
        classes = read(f["Classes/Label"])
        priors = read(f["Classes/Prior"])
        c_names = map(string, classes)
        if read(f["NameType"]) == "Symbol"
            N = Symbol
        else
            N = AbstractString
        end
        info("Reading a model with names of type $N")
        kde_names = Vector{N}()
        discrete_names = Vector{N}()

        for n in names(f[c_names[1]]["Continuous"])
            push!(kde_names, convert(N, n))
        end
        for n in names(f[c_names[1]]["Discrete"])
            push!(discrete_names, convert(N, n))
        end
        classes, priors, kde_names, discrete_names
    end
    return classes, priors, kde_names, discrete_names
end


function to_range{T <: Number}(y::Vector{T})
    min, max = extrema(y)
    dy = (max-min)/(length(y)-1)
    return min:dy:max
end


function load_model{C <: AbstractString}(filename::C)
    classes, priors_vec, kde_names, discrete_names = get_feature_names(filename)
    c_kdes = Dict{Int64, Vector{InterpKDE}}()
    c_discrete = Dict{Int64, Vector{ePDF}}()
    priors = Dict{Int64, Float64}()
    [priors[k]=v for (k,v) in zip(classes, priors_vec)]

    h5open(filename, "r") do f
        for c in classes
            c_discrete[c] = Vector{ePDF}(length(discrete_names))
            c_kdes[c] = Vector{InterpKDE}(length(kde_names))
            for (i, d_name) in enumerate(discrete_names)
                rng = read(f["$c"]["Discrete"]["$d_name"]["range"])
                prob = read(f["$c"]["Discrete"]["$d_name"]["probability"])
                d = Dict{eltype(rng), eltype(prob)}()
                [d[k]=v for (k,v) in zip(rng, prob)]
                c_discrete[c][i] = ePDF(d)
            end
            for (i, c_name) in enumerate(kde_names)
                x = read(f["$c"]["Continuous"]["$c_name"])["x"]
                c_kdes[c][i] = InterpKDE(UnivariateKDE(to_range(x), read(f["$c"]["Continuous"]["$c_name"])["density"]), eps(Float64), InterpLinear)
            end
        end
    end
    return HybridNB{Int64, eltype(kde_names)}(c_kdes, kde_names, c_discrete, discrete_names, classes, priors)
end
