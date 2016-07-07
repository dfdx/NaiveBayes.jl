
import StatsBase.fit
import StatsBase.predict
using Distributions


######################################
#### common naive Bayes functions ####
######################################

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

function fit{C}(m::KernelNB, X::Matrix, y::Vector{C})
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
function sum_log_x_given_c!(class_prob::Vector{Float64}, feature_prob::Vector{Float64}, m::KernelNB, X::Matrix, c)
    for i in 1:size(X, 2)  # for each sample
        for j in 1:size(X, 1)  # for each class
            feature_prob[j] = pdf(m.c_kdes[c][j], X[j, i])
        end
        class_prob[i] = sum(log(feature_prob))
    end
end

"""
    sum_log_x_given_c(feature_prob::Vector{Float64}, m::KernelNB, X::Vector, c)

Returns the class probability (sum(log(P(x|C))))
"""
function sum_log_x_given_c(feature_prob::Vector{Float64}, m::KernelNB, X::Vector, c)
    for j in eachindex(X)
        feature_prob[j] = pdf(m.c_kdes[c][j], X[j])
    end
    class_prob = sum(log(feature_prob))
end

"""
    predict_logprobs(m::KernelNB, X)

Return the log-probabilities for each column of X, where each row is the class
"""
function predict_logprobs{V<:Number}(m::KernelNB, X::Matrix{V})
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
function predict_logprobs{V<:Number}(m::KernelNB, x::Vector{V})
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
    predict_proba(m::KernelNB, X::Vector) -> (class, probability)

Return a tuple of predicted class and log-probability for a single test vector.
"""
function predict_proba{V<:Number}(m::KernelNB, X::Vector{V})
    logprobs = predict_logprobs(m, X)
    classes = collect(keys(m.c_kdes))
    maxprob_idx = indmax(logprobs)
    c = classes[maxprob_idx]
    logprob = logprobs[maxprob_idx]
    return (c, logprob)
end

# Predict kde naive bayes for each column of X
function predict(m::KernelNB, X)
   return [k for (k,v) in predict_proba(m, X)]
end

# TODO remove this once KernelDensity.jl pull request #27 is merged/tagged.
function KernelDensity.InterpKDE{IT<:Grid.InterpType}(k::UnivariateKDE, bc::Number, it::Type{IT}=InterpQuadratic)
    g = CoordInterpGrid(k.x, k.density, bc, it)
    InterpKDE(k, g)
end
