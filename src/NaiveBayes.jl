module NaiveBayes

using Distributions
using HDF5
using KernelDensity
using Grid
using StatsBase

export NBModel,
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
        get_feature_names

include("nbtypes.jl")
include("common.jl")
include("hybrid.jl")
include("gaussian.jl")
include("multinomial.jl")

end
