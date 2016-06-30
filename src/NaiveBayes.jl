
module NaiveBayes

using KernelDensity
using Grid

export NBModel,
       MultinomialNB,
       GaussianNB,
       KernelNB,
       fit,
       predict,
       predict_proba,
       predict_logprobs

include("nbtypes.jl")
include("core.jl")

end
