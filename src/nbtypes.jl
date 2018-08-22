
using Distributions

include("datastats.jl")

"""
Base type for Naive Bayes models.
Inherited classes should have at least following fields:
 c_counts::Dict{C, Int64} - count of ocurrences of each class
 n_obs::Int64             - total number of observations
"""
abstract type NBModel{C} end

#####################################
#####  Multinomial Naive Bayes  #####
#####################################

mutable struct MultinomialNB{C} <: NBModel{C} 
    c_counts::Dict{C, Int64}           # count of ocurrences of each class
    x_counts::Dict{C, Vector{Number}}  # count/sum of occurrences of each var
    x_totals::Vector{Number}           # total occurrences of each var
    n_obs::Int64                       # total number of seen observations
end


"""
Multinomial Naive Bayes classifier

classes : array of objects
    Class names
n_vars : Int64
    Number of variables in observations
alpha : Number (optional, default 1)
    Smoothing parameter. E.g. if alpha equals 1, each variable in each class
    is believed to have 1 observation by default
"""
function MultinomialNB(classes::Vector{C}, n_vars::Int64; alpha=1) where C 
    c_counts = Dict(zip(classes, ones(Int64, length(classes)) * alpha))
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

mutable struct GaussianNB{C} <: NBModel{C}
    c_counts::Dict{C, Int64}           # count of ocurrences of each class
    c_stats::Dict{C, DataStats}        # aggregative data statistics
    gaussians::Dict{C, MvNormal}        # precomputed distribution
    # x_counts::Dict{C, Vector{Number}}  # ?? count/sum of occurrences of each var
    # x_totals::Vector{Number}           # ?? total occurrences of each var
    n_obs::Int64                       # total number of seen observations
end


function GaussianNB(classes::Vector{C}, n_vars::Int64) where C
    c_counts = Dict(zip(classes, zeros(Int64, length(classes))))
    c_stats = Dict(zip(classes, [DataStats(n_vars, 2) for i=1:length(classes)]))
    gaussians = Dict{C, MvNormal}()
    GaussianNB{C}(c_counts, c_stats, gaussians, 0)
end


function Base.show(io::IO, m::GaussianNB)
    print(io, "GaussianNB($(m.c_counts))")
end


#####################################
#####  Hybrid Naive Bayes       #####
#####################################
""" a wrapper around key value pairs for a discrete probability distribution """
struct ePDF{C <: AbstractDict}
    pairs::C
end

""" Constructor of ePDF """
function ePDF(x::AbstractVector{T}) where T <: Integer
    cnts = counts(x)
    ρ = map(Float64, cnts)/sum(cnts)
    ρ[ρ .< eps(Float64)] .= eps(Float64)
    d = Dict{Int, Float64}()
    for (k,v) in zip(StatsBase.span(x), ρ)
        d[k]=v
    end
    return ePDF(d)
end

""" query the ePDF to get the probability of n"""
function probability(P::ePDF, n::Integer)
    if n in keys(P.pairs)
        return P.pairs[n]
    else
        return eps(eltype(values(P.pairs)))
    end
end

"""
Initialize a `HybridNB` model with continuous and/or discrete features

### Constructors
```julia
HybridNB(labels::AbstractVector, kde_names::AbstractVector, discrete_names::AbstractVector)
HybridNB(labels::AbstractVector, kde_names::AbstractVector)
HybridNB(labels::AbstractVector, num_kde::Int, num_discrete::Int)
```
### Arguments
* `labels` : An AbstractVector{Any} of feature labels
* `kde_names` : An AbstractVector{Any} of the names of continuous features
* `discrete_names` : An AbstractVector{Any} of the names of discrete features
* `num_kde` : Number of continuous features
* `num_discrete` : Number of discrete features

"""
struct HybridNB{C <: Integer, N}
    c_kdes::Dict{C, Dict{N, InterpKDE}}
    c_discrete::Dict{C, Dict{N, ePDF}}
    classes::Vector{C}
    priors::Dict{C, Float64}
end


function num_features(m::HybridNB)
    length(m.classes) > 0 || throw("Number of kdes is not defined. There are no valid classes in the model.")
    c = m.classes[1]
    return length(m.c_kdes[c]), length(m.c_discrete[c])
end

num_kdes(m::HybridNB) = num_features(m)[1]
num_discrete(m::HybridNB) = num_features(m)[2]

"""
    HybridNB(labels::Vector{Int64}) -> model_h

    HybridNB(labels::Vector{Int64}, AstractString) -> model_h

A constructor for both types of features
"""
function HybridNB(labels::Vector{C}, ::Type{T}=Symbol) where {C <: Integer, T}
    c_kdes = Dict{C, Dict{T, InterpKDE}}()
    c_discrete = Dict{C, Dict{T, ePDF}}()
    priors = Dict{C, Float64}()
    classes = unique(labels)
    for class in classes
        c_kdes[class] = Dict{T, InterpKDE}()
        c_discrete[class] = Dict{T, ePDF}()
    end
    HybridNB{C, T}(c_kdes, c_discrete, classes, priors)
end

# Initialize with the number of continuous and discrete features
function HybridNB(labels::AbstractVector, num_kde::Int = 0, num_discrete::Int = 0)
    return HybridNB(labels, 1:num_kde, 1:num_discrete)
end

function Base.show(io::IO, m::HybridNB)
    println(io, "HybridNB")
    println(io, "  Classes = $(keys(m.c_kdes))")
end
