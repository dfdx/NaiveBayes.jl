using LinearAlgebra 

"""
    fit(m::HybridNB, f_c::Vector{Vector{Float64}}, f_d::Vector{Vector{Int64}}, labels::Vector{Int64})

Train NB model with discrete and continuous features by estimating P(x⃗|c)
"""
function fit(model::HybridNB, 
	     continuous_features::Dict{N, Vector{T}}, 
	     discrete_features::Dict{N, Vector{U}}, 
	     labels::Vector{C}) where{C, T <: AbstractFloat, U <: Integer, N}

    A = 1.0/float(length(labels))
    for class in model.classes
        inds = findall(labels .== class)
        model.priors[class] = A*float(length(inds))
        for (name, feature) in continuous_features
            f_data = feature[inds]
            model.c_kdes[class][name] = InterpKDE(kde(f_data[isfinite.(f_data)]), eps(Float64),  BSpline(Linear()), OnGrid())
        end
        for (name, feature) in discrete_features
            f_data = feature[inds]
            model.c_discrete[class][name] = ePDF(f_data[isfinite.(f_data)])
        end
    end
    return model
end

"""
    train(HybridNB, continuous, discrete, labels) -> model2
"""
function train(::Type{HybridNB}, 
	       continuous_features::Dict{N, Vector{T}}, 
	       discrete_features::Dict{N, Vector{U}}, 
	       labels::Vector{C}) where{C, T<: AbstractFloat, U <: Integer, N}
    return fit(HybridNB(labels, N), continuous_features, discrete_features, labels)
end


"""
    fit(m::HybridNB, f_c::Matrix{Float64}, labels::Vector{Int64})

Train NB model with continuous features only
"""
function fit(model::HybridNB, 
	     continuous_features::Matrix{T}, 
	     labels::Vector{C}) where{C, T<: AbstractFloat}
    discrete_features = Dict{Symbol, Vector{Int64}}()
    return fit(model, restructure_matrix(continuous_features), discrete_features, labels)
end


"""computes log[P(x⃗ⁿ|c)] ≈ ∑ᵢ log[p(xⁿᵢ|c)] """
function sum_log_x_given_c!(class_prob::Vector{Float64}, 
			    feature_prob::Vector{Float64}, 
			    m::HybridNB, 
			    continuous_features::Dict{N, Vector{T}}, 
			    discrete_features::Dict{N, Vector{U}}, c) where{T <: AbstractFloat, U <: Integer, N}
    for i = 1:num_samples(m, continuous_features, discrete_features)
        for (j, name) in enumerate(keys(continuous_features))
            x_i = continuous_features[name][i]
            feature_prob[j] = isnan(x_i) ? NaN : pdf(m.c_kdes[c][name], x_i)
        end

        for (j, name) in enumerate(keys(discrete_features))
            x_i = discrete_features[name][i]
            feature_prob[num_kdes(m)+j] = isnan(x_i) ? NaN : probability(m.c_discrete[c][name], x_i)
        end
        sel = isfinite.(feature_prob)
        class_prob[i] = sum(log.(feature_prob[sel]))
    end
end


""" compute the number of samples """
function num_samples(m::HybridNB, 
		     continuous_features::Dict{N, Vector{T}}, 
		     discrete_features::Dict{N, Vector{U}}) where{T <: AbstractFloat, U <: Integer, N}
    if length(keys(continuous_features)) > 0
        return length(continuous_features[collect(keys(continuous_features))[1]])
    end
    if length(keys(discrete_features)) > 0
        return length(discrete_features[collect(keys(discrete_features))[1]])
    end
    return 0
end


"""
    predict_logprobs(m::HybridNB, features_c::Vector{Vector{Float64}, features_d::Vector{Vector{Int})

Return the log-probabilities for each column of X, where each row is the class
"""
function predict_logprobs(m::HybridNB, 
			  continuous_features::Dict{N, Vector{T}}, 
			  discrete_features::Dict{N, Vector{U}}) where{T <: AbstractFloat, U <: Integer, N}
    n_samples = num_samples(m, continuous_features, discrete_features)
    log_probs_per_class = zeros(length(m.classes) ,n_samples)
    feature_prob = Vector{Float64}(undef, num_kdes(m) + num_discrete(m))
    for (i, c) in enumerate(m.classes)
        class_prob = Vector{Float64}(undef, n_samples)
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
function predict_proba(m::HybridNB, 
		       continuous_features::Dict{N, Vector{T}}, discrete_features::Dict{N, Vector{U}}) where{T <: AbstractFloat, U <: Integer, N}
    logprobs = predict_logprobs(m, continuous_features, discrete_features)
    n_samples = num_samples(m, continuous_features, discrete_features)
    predictions = Array{Tuple{eltype(m.classes), Float64}}(undef, n_samples)
    for i = 1:n_samples
        maxprob_idx = argmax(logprobs[:, i])
        c = m.classes[maxprob_idx]
        logprob = logprobs[maxprob_idx, i]
        predictions[i] = (c, logprob)
    end
    return predictions
end

""" Predict kde naive bayes for continuos featuers only""" # TODO: remove this
function predict(m::HybridNB, X::Matrix{T}) where {T <: Number}
    eltype(X) <: AbstractFloat || throw("Continuous features must be floats!")
    return predict(m, restructure_matrix(X), Dict{Symbol, Vector{Int}}())
end

"""
    predict(m::HybridNB, f_c::Vector{Vector{Float64}}, f_d::Vector{Vector{Int64}}) -> labels

Predict hybrid naive bayes for continuos featuers only
"""
function predict(m::HybridNB, 
		 continuous_features::Dict{N, Vector{T}}, 
		 discrete_features::Dict{N, Vector{U}}) where  {T <: AbstractFloat, U <: Integer, N}
    return [k for (k,v) in predict_proba(m, continuous_features, discrete_features)]
end

# TODO Temporary fix to add extrapolation when outside (Remove once PR in KernelDensity.jl is merged)
import KernelDensity: InterpKDE
import Interpolations: ExtrapDimSpec
function InterpKDE(kde::UnivariateKDE, extrap::Union{ExtrapDimSpec, Number}, opts...)
    itp_u = interpolate(kde.density, opts...)
    itp_u = extrapolate(itp_u, extrap)
    itp = Interpolations.scale(itp_u, kde.x)
    InterpKDE{typeof(kde),typeof(itp)}(kde, itp)
end
InterpKDE(kde::UnivariateKDE) = InterpKDE(kde, NaN, BSpline(Quadratic(Line())), OnGrid())


function write_model(model::HybridNB, filename::S) where {S <: AbstractString}
    h5open(filename, "w") do f
        name_type = eltype(keys(model.c_kdes[model.classes[1]]))
        f["NameType"] = "$name_type"
        @info("Writing a model with names of type $name_type")
        f["Labels"] = model.classes
        for c in model.classes
            grp = g_create(f, "$c")
            grp["Prior"] = model.priors[c]
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
    @info("Writing HybridNB model to file $filename")
end


function to_range(y::Vector{T}) where {T <: Number}
    min, max = extrema(y)
    dy = (max-min)/(length(y)-1)
    return min:dy:max
end


function load_model(filename::S) where {S <: AbstractString}
    model = h5open(filename, "r") do f
        N = read(f["NameType"]) == "Symbol" ? Symbol : AbstractString
        fnc = N == AbstractString ? string : Symbol
        classes = read(f["Labels"])
        C = eltype(classes)
        priors = Dict{C, Float64}()
        kdes = Dict{C, Dict{N, InterpKDE}}()
        discrete = Dict{C, Dict{N, ePDF}}()
        for c in classes
            priors[c] = read(f["$c"]["Prior"])
            kdes[c] = Dict{N, InterpKDE}()
            for (name, dist) in read(f["$c"]["Continuous"])
                kdes[c][fnc(name)] = InterpKDE(UnivariateKDE(to_range(dist["x"]), dist["density"]), eps(Float64), BSpline(Linear()), OnGrid())
            end
            discrete[c] = Dict{N, ePDF}()
            for (name, dist) in read(f["$c"]["Discrete"])
                rng = dist["range"]
                prob = dist["probability"]
                d = Dict{eltype(rng), eltype(prob)}()
                [d[k]=v for (k,v) in zip(rng, prob)]
                discrete[c][fnc(name)] = ePDF(d)
            end
        end
        return HybridNB{C, N}(kdes, discrete, classes, priors)
    end
    return model
end
