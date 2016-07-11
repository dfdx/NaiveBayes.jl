
import StatsBase.fit
import StatsBase.predict
import StatsBase.span
using Distributions
using StatsBase

######################################
#### common naive Bayes functions ####
######################################

""" convert a vector of vector into a matrix """
function to_matrix{T <: Number}(V::Vector{Vector{T}})
    n_lines = length(V)
    n_lines < 1  && throw("Empty vector")
    X = zeros(n_lines, length(V[1]))
    for i=1:n_lines
        X[i, :] = V[i]
    end
    return X
end

""" convert a matrix to vector of vectors"""
function from_matrix{T <: Number}(M::Matrix{T})
    d, n = size(M)
    V = Vector{Vector{eltype(M)}}(d)
    for i=1:d
        V[i] = vec(M[i, :])
    end
    return V
end


function ensure_data_size(X, y)
    @assert(size(X, 2) == length(y),
            "Number of observations in X ($(size(X, 2))) is not equal to " *
            "number of class labels in y ($(length(y)))")
end

function logprob_c{C}(m::NBModel, c::C)
    return log(m.c_counts[c] / m.n_obs)
end

"""Predict log probabilities for all classes"""
function predict_logprobs{V<:Number}(m::NBModel, x::Vector{V})
    C = eltype(keys(m.c_counts))
    logprobs = Dict{C, Float64}()
    for c in keys(m.c_counts)
        logprobs[c] = logprob_c(m, c) + logprob_x_given_c(m, x, c)
    end
    return keys(logprobs), values(logprobs)
end

"""Predict log probabilities for all classes"""
function predict_logprobs{V<:Number}(m::NBModel, X::Matrix{V})
    C = eltype(keys(m.c_counts))
    logprobs_per_class = Dict{C, Vector{Float64}}()
    for c in keys(m.c_counts)
        logprobs_per_class[c] = logprob_c(m, c) + logprob_x_given_c(m, X, c)
    end
    return (collect(keys(logprobs_per_class)),
            hcat(collect(values(logprobs_per_class))...)')
end

"""Predict logprobs, return tuples of predicted class and its logprob"""
function predict_proba{V<:Number}(m::NBModel, X::Matrix{V})
    C = eltype(keys(m.c_counts))
    classes, logprobs = predict_logprobs(m, X)
    predictions = Array(Tuple{C, Float64}, size(X, 2))
    for j=1:size(X, 2)
        maxprob_idx = indmax(logprobs[:, j])
        c = classes[maxprob_idx]
        logprob = logprobs[maxprob_idx, j]
        predictions[j] = (c, logprob)
    end
    return predictions
end

function predict{V<:Number}(m::NBModel, X::Matrix{V})
    return [k for (k,v) in predict_proba(m, X)]
end


#####################################
#####  Multinomial Naive Bayes  #####
#####################################

function fit{C}(m::MultinomialNB, X::Matrix{Int64}, y::Vector{C})
    ensure_data_size(X, y)
    for j=1:size(X, 2)
        c = y[j]
        m.c_counts[c] += 1
        m.x_counts[c] .+= X[:, j]
        m.x_totals += X[:, j]
        m.n_obs += 1
    end
    return m
end

"""Calculate log P(x|C)"""
function logprob_x_given_c{C}(m::MultinomialNB, x::Vector{Int64}, c::C)
    x_priors_for_c = m.x_counts[c] ./ m.x_totals
    x_probs_given_c = x_priors_for_c .^ x
    logprob = sum(log(x_probs_given_c))
    return logprob
end

"""Calculate log P(x|C)"""
function logprob_x_given_c{C}(m::MultinomialNB, X::Matrix{Int64}, c::C)
    x_priors_for_c = m.x_counts[c] ./ m.x_totals
    x_probs_given_c = x_priors_for_c .^ X
    logprob = sum(log(x_probs_given_c), 1)
    return squeeze(logprob, 1)
end

#####################################
######  Gaussian Naive Bayes  #######
#####################################

function fit{C}(m::GaussianNB, X::Matrix{Float64}, y::Vector{C})
    ensure_data_size(X, y)
    # updatestats(m.dstats, X)
    # m.gaussian = MvNormal(mean(m.dstats), cov(m.dstats))
    # m.n_obs = m.dstats.n_obs
    n_vars = size(X, 1)
    for j=1:size(X, 2)
        c = y[j]
        m.c_counts[c] += 1
        updatestats(m.c_stats[c], reshape(X[:, j], n_vars, 1))
        # m.x_counts[c] .+= X[:, j]
        # m.x_totals += X[:, j]
        m.n_obs += 1
    end
    # precompute distributions for each class
    for c in keys(m.c_counts)
        m.gaussians[c] = MvNormal(mean(m.c_stats[c]), cov(m.c_stats[c]))
    end
    return m
end


"""Calculate log P(x|C)"""
function logprob_x_given_c{C}(m::GaussianNB, x::Vector{Float64}, c::C)
    return logpdf(m.gaussians[c], x)
end


"""Calculate log P(x|C)"""
function logprob_x_given_c{C}(m::GaussianNB, X::Matrix{Float64}, c::C)
    ## x_priors_for_c = m.x_counts[c] ./ m.x_totals
    ## x_probs_given_c = x_priors_for_c .^ x
    ## logprob = sum(log(x_probs_given_c))
    ## return logprob
    return logpdf(m.gaussians[c], X)
end

#####################################
#####  Kernel Naive Bayes       #####
#####################################

"""
    fit(m::HybridNB, f_c::Vector{Vector{Float64}}, f_d::Vector{Vector{Int64}}, labels::Vector{Int64})

Train NB model with discrete and continuous features
"""
function fit{C, T<: AbstractFloat, U<:Int}(model::HybridNB, continuous_features::Vector{Vector{T}}, discrete_features::Vector{Vector{U}}, labels::Vector{C})
    for class in model.classes
        inds = find(labels .== class)
        for (j, feature) in enumerate(continuous_features)
            model.c_kdes[class][j] = InterpKDE(kde(feature[inds]), eps(Float64), InterpLinear)
        end
        for (j, feature) in enumerate(discrete_features)
            model.c_discrete[class][j] = ePDF(feature[inds])
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
    return fit(model, from_matrix(continuous_features), discrete_features, labels)
end


function fit{C}(m::KernelNB, X::Matrix, y::Vector{C})
    warn("fit method for KernelNB is deprecated. Use HybridNB instead.")
    ensure_data_size(X, y)
    unique_classes = unique(y)
    for j in 1:size(X, 1)
        for class in unique_classes
            inds = find(class .== y)
            m.c_kdes[class][j] = InterpKDE(kde(vec(X[j, inds])), eps(Float64), InterpLinear)
        end
    end
    return m
end


"""
    sum_log_x_given_c!(class_prob::Vector{Float64}, feature_prob::Vector{Float64}, m::KernelNB, X::Matrix, c)

Updates input vector `class_prob` with the sum(log(P(x|C)))
"""
function sum_log_x_given_c!(class_prob::Vector{Float64}, feature_prob::Vector{Float64}, m::KernelNB, X::Matrix, c)# TODO: depricate
    for i in 1:size(X, 2)  # for each sample
        for j in 1:size(X, 1)  # for each feature
            feature_prob[j] = pdf(m.c_kdes[c][j], X[j, i])
        end
        class_prob[i] = sum(log(feature_prob))
    end
end


"""computes log[P(x⃗ⁿ|c)] ≈ ∑ᵢ log[p(xⁿᵢ|c)] """
function sum_log_x_given_c!{T <: AbstractFloat, U <: Int}(class_prob::Vector{Float64}, feature_prob::Vector{Float64}, m::HybridNB, continuous_features::Vector{Vector{T}}, discrete_features::Vector{Vector{U}}, c)
    for i = 1:num_samples(m, continuous_features, discrete_features)
        for j = 1:m.num_kdes
            feature_prob[j] = pdf(m.c_kdes[c][j], continuous_features[j][i])
        end
        for j = 1:m.num_discrete
            feature_prob[m.num_kdes+j] = probability(m.c_discrete[c][j], discrete_features[j][i])
        end
        class_prob[i] = sum(log(feature_prob))
    end
end


"""
    sum_log_x_given_c(feature_prob::Vector{Float64}, m::KernelNB, X::Vector, c)

Returns the class probability (sum(log(P(x|C))))
"""
function sum_log_x_given_c(feature_prob::Vector{Float64}, m::KernelNB, X::Vector, c) # TODO: depricate
    for j in eachindex(X)
        feature_prob[j] = pdf(m.c_kdes[c][j], X[j])
    end
    class_prob = sum(log(feature_prob))
end


""" compute the number of samples """
function num_samples{T <: AbstractFloat, U <: Int}(m::HybridNB, continuous_features::Vector{Vector{T}}, discrete_features::Vector{Vector{U}}) # TODO: this is a bit strange
    if m.num_kdes > m.num_discrete
        n_samples = length(continuous_features[1])
    else
        n_samples = length(discrete_features[1])
    end
    return n_samples
end

"""
    predict_logprobs(m::HybridNB, features_c::Vector{Vector{Float64}, features_d::Vector{Vector{Int})

Return the log-probabilities for each column of X, where each row is the class
"""#FIXME me
function predict_logprobs{T <: AbstractFloat, U <: Int}(m::HybridNB, continuous_features::Vector{Vector{T}}, discrete_features::Vector{Vector{U}})
    n_samples = num_samples(m, continuous_features, discrete_features)
    log_probs_per_class = zeros(length(m.classes) ,n_samples)
    feature_prob = Vector{Float64}(m.num_kdes + m.num_discrete)
    for (i, c) in enumerate(m.classes)
        class_prob = Vector{Float64}(n_samples)
        sum_log_x_given_c!(class_prob, feature_prob, m, continuous_features, discrete_features, c)
        log_probs_per_class[i, :] = class_prob
    end
    return log_probs_per_class
end


"""
    predict_logprobs(m::KernelNB, X)

Return the log-probabilities for each column of X, where each row is the class
"""
function predict_logprobs{V<:Number}(m::KernelNB, X::Matrix{V})
    warn("method predict_logprobs for KernelNB is depricated")
    log_probs_per_class = Dict{eltype(keys(m.c_kdes)), Vector{Float64}}()
    feature_prob = Vector{Float64}(m.n_vars)
    n_samples = size(X, 2)
    for c in keys(m.c_kdes)
        class_prob = Vector{Float64}(n_samples)
        sum_log_x_given_c!(class_prob, feature_prob, m, X, c)
        log_probs_per_class[c] = class_prob
    end
    return hcat(collect(values(log_probs_per_class))...)'
end

# Predict log-probabilities for a vector
function predict_logprobs{V<:Number}(m::KernelNB, x::Vector{V}) #TODO: is that really nedded?
    logprobs = Dict{eltype(keys(m.c_kdes)), Float64}()
    feature_prob = Vector{Float64}(m.n_vars)
    for c in keys(m.c_kdes)
        logprobs[c] = sum_log_x_given_c(feature_prob, m, x, c)
    end
    return collect(values(logprobs))
end

"""
    predict_proba{V<:Number}(m::KernelNB, X::Matrix{V})

Predict log-probabilities for the input column vectors.
Returns tuples of predicted class and its log-probability estimate.
"""
function predict_proba{V<:Number}(m::KernelNB, X::Matrix{V})
    warn("method predict_proba for KernelNB is depricated")
    logprobs = predict_logprobs(m, X)
    classes = collect(keys(m.c_kdes))
    n_samples = size(X, 2)
    predictions = Array(Tuple{eltype(classes), Float64}, n_samples)
    for i in 1:n_samples
        maxprob_idx = indmax(logprobs[:, i])
        c = classes[maxprob_idx]
        logprob = logprobs[maxprob_idx, i]
        predictions[i] = (c, logprob)
    end
    return predictions
end


"""
    predict_proba{V<:Number}(m::HybridNB, f_c::Vector{Vector{Float64}}, f_d::Vector{Vector{Int64}})

Predict log-probabilities for the input features.
Returns tuples of predicted class and its log-probability estimate.
"""
function predict_proba{T <: AbstractFloat, U <: Int}(m::HybridNB, continuous_features::Vector{Vector{T}}, discrete_features::Vector{Vector{U}})
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


"""
    predict_proba(m::KernelNB, X::Vector) -> (class, probability)

Return a tuple of predicted class and log-probability for a single test vector.
"""
function predict_proba{V<:Number}(m::KernelNB, X::Vector{V}) # what is that for?
    logprobs = predict_logprobs(m, X)
    classes = collect(keys(m.c_kdes))
    maxprob_idx = indmax(logprobs)
    c = classes[maxprob_idx]
    logprob = logprobs[maxprob_idx]
    return (c, logprob)
end

# Predict kde naive bayes for each column of X
function predict(m::KernelNB, X)
    warn("predict method for KernelNB is depricated. Use HybridNB instead.")
    return [k for (k,v) in predict_proba(m, X)]
end

""" Predict kde naive bayes for continuos featuers only"""
function predict{T <: Number}(m::HybridNB, X::Matrix{T})
    eltype(X) <: AbstractFloat || throw("Continuous features must be floats!")
    return predict(m, from_matrix(X), Vector{Vector{Int}}())
end

"""
    predict(m::HybridNB, f_c::Vector{Vector{Float64}}, f_d::Vector{Vector{Int64}}) -> labels

Predict hybrid naive bayes for continuos featuers only
"""
function predict{T <: AbstractFloat, U <: Int}(m::HybridNB, continuous_features::Vector{Vector{T}}, discrete_features::Vector{Vector{U}})
    return [k for (k,v) in predict_proba(m, continuous_features, discrete_features)]
end

# TODO remove this once KernelDensity.jl pull request #27 is merged/tagged.
function KernelDensity.InterpKDE{IT<:Grid.InterpType}(k::UnivariateKDE, bc::Number, it::Type{IT}=InterpQuadratic)
    g = CoordInterpGrid(k.x, k.density, bc, it)
    InterpKDE(k, g)
end
