function fit(m::MultinomialNB, X::Matrix{Int64}, y::Vector{C}) where C
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
function logprob_x_given_c(m::MultinomialNB, x::Vector{Int64}, c::C) where C
    x_priors_for_c = m.x_counts[c] ./ m.x_totals
    x_probs_given_c = x_priors_for_c .^ x
    logprob = sum(log(x_probs_given_c))
    return logprob
end

"""Calculate log P(x|C)"""
function logprob_x_given_c(m::MultinomialNB, X::Matrix{Int64}, c::C) where C
    x_priors_for_c = m.x_counts[c] ./ m.x_totals
    x_probs_given_c = x_priors_for_c .^ X
    logprob = sum(log.(x_probs_given_c), dims=1)
    return dropdims(logprob, dims=1)
end
