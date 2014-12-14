
include("datastats.jl")

abstract NBModel{C}

#####################################
#####  Multinomial Naive Bayes  #####
#####################################

type MultinomialNB{C} <: NBModel
    c_counts::Dict{C, Int64}           # count of ocurrences of each class
    x_counts::Dict{C, Vector{Number}}  # count/sum of occurrences of each var
    x_totals::Vector{Number}           # total occurrences of each var
    n_obs::Int64                       # total number of seen observations
end


# Multinomial Naive Bayes classifier
#
# classes : array of objects
#     Class names
# n_vars : Int64
#     Number of variables in observations
# alpha : Number (optional, default 1)
#     Smoothing parameter. E.g. if alpha equals 1, each variable in each class
#     is believed to have 1 observation by default
function MultinomialNB{C}(classes::Vector{C}, n_vars::Int64; alpha=1)
    c_counts = Dict(classes, ones(Int64, length(classes)) * alpha)
    x_counts = Dict{C, Vector{Int64}}()
    for c in classes
        x_counts[c] = ones(Int64, n_vars) * alpha
    end
    x_totals = ones(Float64, n_vars) * alpha * length(c_counts)
    MultinomialNB{C}(c_counts, x_counts, x_totals, sum(x_totals))
end

function Base.show(io::IO, m::MultinomialNB)
    print(io, "MultinomialNB($(m.c_counts))")
end


#####################################
######  Gaussian Naive Bayes  #######
#####################################

type GaussianNB{C} <: NBModel
    c_counts::Dict{C, Int64}           # count of ocurrences of each class
    dstats::DataStats                  # aggregative data statistics   
    x_counts::Dict{C, Vector{Number}}  # ?? count/sum of occurrences of each var
    x_totals::Vector{Number}           # ?? total occurrences of each var
    n_obs::Int64                       # total number of seen observations
end

function GaussianNB{C}(classes::Vector{C}, n_vars::Int64; alpha=0)
    c_counts = Dict(classes, ones(Int64, length(classes)) * alpha)
    x_counts = Dict{C, Vector{Int64}}()
    for c in classes
        x_counts[c] = ones(Int64, n_vars) * alpha
    end
    x_totals = ones(Float64, n_vars) * alpha * length(c_counts)
    GaussianNB{C}(c_counts, x_counts, x_totals, sum(x_totals))
end
