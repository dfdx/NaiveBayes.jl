
module NaiveBayes

using KernelDensity
using Grid

export NBModel,
        MultinomialNB,
        GaussianNB,
        KernelNB,
        HybridNB,
        fit,
        predict,
        predict_proba,
        predict_logprobs,
        from_matrix,
        to_matrix

include("nbtypes.jl")
include("core.jl")

end
