
abstract NBModel{C}

type MultinomialNB{C} <: NBModel
    c_counts::Dict{C, Int64}           # count of ocurrences of each class
    x_counts::Dict{C, Vector{Number}}  # count/sum of occurrences of each var
    x_totals::Vector{Number}           # total occurrences of each var
    n_obs::Int64                       # total number of seen observations
    function MultinomialNB(classes::Vector{C}, n_vars::Int64; smoothing=1)
        c_counts = Dict(classes, zeros(Int64, length(classes)))        
        x_counts = Dict{C, Vector{Int64}}()
        for c in classes
            x_counts[c] = ones(Int64, n_vars) * smoothing
        end
        x_totals = ones(Float64, n_vars) * smoothing * length(c_counts)
        new(c_counts, x_counts, x_totals, 0)
    end
end
