
using Distributions
using StatsBase

include("datastats.jl")

"""
Base type for Naive Bayes models.
Inherited classes should have at least following fields:
 c_counts::Dict{C, Int64} - count of ocurrences of each class
 n_obs::Int64             - total number of observations
"""
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
function MultinomialNB{C}(classes::Vector{C}, n_vars::Int64; alpha=1)
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

type GaussianNB{C} <: NBModel
    c_counts::Dict{C, Int64}           # count of ocurrences of each class
    c_stats::Dict{C, DataStats}        # aggregative data statistics
    gaussians::Dict{C, MvNormal}        # precomputed distribution
    # x_counts::Dict{C, Vector{Number}}  # ?? count/sum of occurrences of each var
    # x_totals::Vector{Number}           # ?? total occurrences of each var
    n_obs::Int64                       # total number of seen observations
end


function GaussianNB{C}(classes::Vector{C}, n_vars::Int64)
    c_counts = Dict(zip(classes, zeros(Int64, length(classes))))
    c_stats = Dict(zip(classes, [DataStats(n_vars, 2) for i=1:length(classes)]))
    gaussians = Dict{C, MvNormal}()
    GaussianNB{C}(c_counts, c_stats, gaussians, 0)
end


function Base.show(io::IO, m::GaussianNB)
    print(io, "GaussianNB($(m.c_counts))")
end

#####################################
#####  Kernel Naive Bayes       #####
#####################################

immutable KernelNB{C}
    c_kdes::Dict{C, Vector{InterpKDE}}
    n_vars::Int
end

function KernelNB{C}(classes::Vector{C}, n_vars::Int)
    classes = unique(classes)
    c_kdes = Dict{C, Vector{InterpKDE}}()
    for class in classes
        c_kdes[class] = Vector{InterpKDE}(n_vars)
    end
    KernelNB{C}(c_kdes, n_vars)
end

function Base.show(io::IO, m::KernelNB)
    println(io, "KernelNB")
    println(io, "  Classes = $(keys(m.c_kdes))")
    print(io, "  Number of variables = $(m.n_vars)")
end

#####################################
#####  Hybrid Naive Bayes       #####
#####################################

type ePDF{R <: AbstractArray, T <: AbstractFloat}
    n::R
    probability::Vector{T}
end

function ePDF{T <: Int}(x::AbstractArray{T})
    cnts = counts(x)
    ρ = map(Float64, cnts)/sum(cnts)
    ρ[ρ .< eps()] = eps()
    return ePDF(StatsBase.span(x), ρ)
end


""" A Naive Bayes model for both continuoes and discrete features"""
immutable HybridNB{C}
    c_kdes::Dict{C, Vector{InterpKDE}}
    num_kdes::Int
    c_discrete::Dict{C, Vector{ePDF}}
    num_discrete::Int # it would be nice to have the number of classes and the number of training examples for each class
    class_prior::Vector{Float64}
end


function HybridNB{C, T <: Integer}(labels::Vector{C}, num_kdes::T, num_discrete::T)
    c_kdes = Dict{C, Vector{InterpKDE}}()
    c_discrete = Dict{C, Vector{ePDF}}()
    classes = unique(labels)
    priors = map(Float64, counts(labels))/length(labels)
    for class in classes
        c_kdes[class] = Vector{InterpKDE}(num_kdes)
        c_discrete[class] = Vector{ePDF}(num_discrete)
    end
    HybridNB{C}(c_kdes, num_kdes, c_discrete, num_discrete, priors)
end


function Base.show(io::IO, m::HybridNB)
    println(io, "HybridNB")
    println(io, "  Classes = $(keys(m.c_kdes))")
    println(io, "  Number of continiues features = $(m.num_kdes)")
    println(io, "  Number of discrete features = $(m.num_discrete)")
end
