
import StatsBase.fit
import StatsBase.predict


######################################
#### common naive Bayes functions ####
######################################

function logprob_c{C}(m::NBModel, c::C)
    return m.c_counts[c] / m.n_obs
end

# predict log probabilities for all classes
function predict_logprobs{V<:Number}(m::NBModel, x::Vector{V})
    C = eltype(m.c_counts)[1]
    logprobs = Dict{C, Float64}()
    for c in keys(m.c_counts)
        logprobs[c] = logprob_c(m, c) + logprob_x_given_c(m, x, c)
    end
    return keys(logprobs), values(logprobs)
end

# predict log probabilities for all classes
function predict_logprobs{V<:Number}(m::NBModel, X::Matrix{V})
    C = eltype(m.c_counts)[1]
    logprobs_per_class = Dict{C, Vector{Float64}}()
    for c in keys(m.c_counts)
        logprobs_per_class[c] = logprob_c(m, c) + logprob_x_given_c(m, X, c)
    end
    return (collect(keys(logprobs_per_class)),
            hcat(collect(values(logprobs_per_class))...)')
end

# preditct logprobs, return tuples of predicted class and its logprob
function predict_proba{V<:Number}(m::NBModel, X::Matrix{V})
    C = eltype(m.c_counts)[1]
    classes, logprobs = predict_logprobs(m, X)
    predictions = Array((C, Float64), size(X, 2))
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

# Calculate log P(x|C)
function logprob_x_given_c{C}(m::MultinomialNB, x::Vector{Int64}, c::C)
    x_priors_for_c = m.x_counts[c] ./ m.x_totals
    x_probs_given_c = x_priors_for_c .^ x
    logprob = sum(log(x_probs_given_c))
    return logprob
end

# Calculate log P(x|C)
function logprob_x_given_c{C}(m::MultinomialNB, X::Matrix{Int64}, c::C)
    x_priors_for_c = m.x_counts[c] ./ m.x_totals
    x_probs_given_c = x_priors_for_c .^ X
    logprob = sum(log(x_probs_given_c), 1)
    return squeeze(logprob, 1)
end


function fit{C}(m::MultinomialNB, X::Matrix{Int64}, y::Vector{C})
    @assert(size(X, 2) == length(y),
            "Number of observations in X ($(size(X, 2))) is not equal to " *
            "number of class labels in y ($(length(y)))")
    for j=1:size(X, 2)
        c = y[j]
        m.c_counts[c] += 1
        m.x_counts[c] .+= X[:, j]
        m.x_totals += X[:, j]
        m.n_obs += 1
    end
    return m
end


#####################################
######  Gaussian Naive Bayes  #######
#####################################

# Calculate log P(x|C)
function logprob_x_given_c{C}(m::MultinomialNB, x::Vector{Int64}, c::C)
    x_priors_for_c = m.x_counts[c] ./ m.x_totals
    x_probs_given_c = x_priors_for_c .^ x
    logprob = sum(log(x_probs_given_c))
    return logprob
end
