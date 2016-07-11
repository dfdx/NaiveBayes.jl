######################################
#### common naive Bayes functions ####
######################################

"""
    to_matrix(V::Vector{Vector}}) -> M::Matrix

convert a vector of vector into a matrix
"""
function to_matrix{T <: Number}(V::Vector{Vector{T}})
    n_lines = length(V)
    n_lines < 1  && throw(ArgumentError("Empty input"))
    X = zeros(n_lines, length(V[1]))
    for i=1:n_lines
        X[i, :] = V[i]
    end
    return X
end

"""
    restructure_matrix(M::Matrix) -> V::Vector{Vector}

Restructure a matrix as vector of vectors
"""
function restructure_matrix{T <: Number}(M::Matrix{T})
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
