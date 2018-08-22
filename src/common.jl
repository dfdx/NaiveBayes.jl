######################################
#### common naive Bayes functions ####
######################################
"""
    to_matrix(D::Dict{Symbol, Vector}}) -> M::Matrix

convert a dictionary of vectors into a matrix
"""
function to_matrix(V::Dict{N, Vector{T}}) where {T <: Number, N}
    n_features = length(V)
    n_features < 1  && throw(ArgumentError("Empty input"))
    X = zeros(n_features, length(V[collect(keys(V))[1]]))
    for (i, f) in enumerate(values(sort(collect(V))))
	    X[i, :] = f[2]
    end
    return X
end


"""
    restructure_matrix(M::Matrix) -> V::Dict{Symbol, Vector}

Restructure a matrix as vector of vectors
"""
function restructure_matrix(M::Matrix{T}) where {T <: Number}
    d, n = size(M)
    V = Dict{Symbol, Vector{eltype(M)}}()
    for i=1:d
        V[Symbol("x$i")] = vec(M[i, :]) 
    end
    return V
end


function ensure_data_size(X, y)
    @assert(size(X, 2) == length(y),
            "Number of observations in X ($(size(X, 2))) is not equal to " *
            "number of class labels in y ($(length(y)))")
end

function logprob_c(m::NBModel, c::C) where C
    return log(m.c_counts[c] / m.n_obs)
end

"""Predict log probabilities for all classes"""
function predict_logprobs(m::NBModel, x::Vector{V}) where {V<:Number}
    C = eltype(keys(m.c_counts))
    logprobs = Dict{C, Float64}()
    for c in keys(m.c_counts)
        logprobs[c] = logprob_c(m, c) + logprob_x_given_c(m, x, c)
    end
    return keys(logprobs), values(logprobs)
end

"""Predict log probabilities for all classes"""
function predict_logprobs(m::NBModel, X::Matrix{V}) where {V<:Number}
    C = eltype(keys(m.c_counts))
    logprobs_per_class = Dict{C, Vector{Float64}}()
    for c in keys(m.c_counts)
        logprobs_per_class[c] = logprob_c(m, c) .+ logprob_x_given_c(m, X, c)
    end
    return (collect(keys(logprobs_per_class)),
            hcat(collect(values(logprobs_per_class))...)')
end

"""Predict logprobs, return tuples of predicted class and its logprob"""
function predict_proba(m::NBModel, X::Matrix{V}) where {V<:Number}
    C = eltype(keys(m.c_counts))
    classes, logprobs = predict_logprobs(m, X)
    predictions = Array{Tuple{C, Float64}}(undef, size(X, 2))
    for j=1:size(X, 2)
        maxprob_idx = argmax(logprobs[:, j])
        c = classes[maxprob_idx]
        logprob = logprobs[maxprob_idx, j]
        predictions[j] = (c, logprob)
    end
    return predictions
end

function predict(m::NBModel, X::Matrix{V}) where {V<:Number}
    return [k for (k,v) in predict_proba(m, X)]
end
