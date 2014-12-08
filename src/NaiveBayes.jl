
module NaiveBayes

export NBModel,
       MultinomialNB,
       fit,
       predict,
       predict_proba

include("nbtypes.jl")
include("core.jl")

end
