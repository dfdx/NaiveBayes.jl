
module NaiveBayes

export NBModel,
       MultinomialNB,
       fit,
       predict,
       predict_proba

require("nbtypes.jl")
include("core.jl")

end
