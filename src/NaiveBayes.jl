module NaiveBayes

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


include("core.jl")

end
