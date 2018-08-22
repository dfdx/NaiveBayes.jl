module NaiveBayes

using Distributions
using HDF5
using KernelDensity
using Interpolations
using LinearAlgebra
using StatsBase
import StatsBase: fit, predict

export  NBModel,
        MultinomialNB,
        GaussianNB,
        KernelNB,
        HybridNB,
        fit,
        predict,
        predict_proba,
        predict_logprobs,
        restructure_matrix,
        to_matrix,
        write_model,
        load_model,
        get_feature_names,
        train

include("nbtypes.jl")
include("common.jl")
include("hybrid.jl")
include("gaussian.jl")
include("multinomial.jl")

end
