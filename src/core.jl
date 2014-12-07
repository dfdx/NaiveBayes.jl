
import StatsBase.fit
import StatsBase.predict


function Base.show(io::IO, m::MultinomialNB)
    print(io, "NBModel($(m.c_counts))")
end

function logprob_c{C}(m::NBModel, c::C)
    return m.c_counts[c] / m.n_obs
end

# Calculate log P(x|C)
function logprob_x_given_c{C, V<:Number}(m::NBModel, x::Vector{V}, c::C)
    x_priors_for_c = m.x_counts[c] ./ m.x_totals
    x_probs_given_c = x_priors_for_c .^ x
    logprob = sum(log(x_probs_given_c))
    return logprob
end


# Calculate log P(x|C)
function logprob_x_given_c{C, V<:Number}(m::NBModel, X::Matrix{V}, c::C)
    x_priors_for_c = m.x_counts[c] ./ m.x_totals
    x_probs_given_c = x_priors_for_c .^ X
    logprob = sum(log(x_probs_given_c), 1)
    return squeeze(logprob, 1)
end


function fit{C, V<:Number}(m::NBModel, X::Matrix{V}, y::Vector{C})
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

# predict log probabilities for all classes
function predict_logprobs{C, V<:Number}(m::NBModel{C}, x::Vector{V})
    logprobs = Dict{C, Float64}()
    for c in keys(m.c_counts)
        logprobs[c] = logprob_c(m, c) + logprob_x_given_c(m, x, c)
    end
    return keys(logprobs), values(logprobs)
end

# predict log probabilities for all classes
function predict_logprobs{C, V<:Number}(m::NBModel{C}, X::Matrix{V})
    logprobs_per_class = Dict{C, Vector{Float64}}()
    for c in keys(m.c_counts)
        logprobs_per_class[c] = logprob_c(m, c) + logprob_x_given_c(m, X, c)
    end
    return (collect(keys(logprobs_per_class)),
            hcat(collect(values(logprobs_per_class))...)')
end


function predict_proba{C, V<:Number}(m::NBModel{C}, X::Matrix{V})
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

function predict{C, V<:Number}(m::NBModel{C}, X::Matrix{V})
    return collect(keys(precict_proba(m, X)))
end

