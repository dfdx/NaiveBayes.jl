
module NaiveBayes

export NBModel,
       MultinomialNB,
       GaussianNB,
       fit,
       predict,
       predict_proba,
       predict_logprobs

include("nbtypes.jl")
include("core.jl")

end
