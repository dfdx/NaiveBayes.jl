function fit(m::GaussianNB, X::Matrix{Float64}, y::Vector{C}) where C
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
function logprob_x_given_c(m::GaussianNB, x::Vector{Float64}, c::C) where C
    return logpdf(m.gaussians[c], x)
end


"""Calculate log P(x|C)"""
function logprob_x_given_c(m::GaussianNB, X::Matrix{Float64}, c::C) where C
    ## x_priors_for_c = m.x_counts[c] ./ m.x_totals
    ## x_probs_given_c = x_priors_for_c .^ x
    ## logprob = sum(log(x_probs_given_c))
    ## return logprob
    return logpdf(m.gaussians[c], X)
end
