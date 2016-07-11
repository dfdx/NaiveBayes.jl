NaiveBayes.jl
=============

[![Build Status](https://travis-ci.org/dfdx/NaiveBayes.jl.svg)](https://travis-ci.org/dfdx/NaiveBayes.jl)

Naive Bayes classifier. Currently 4 types of NB are supported:

 * **MultinomialNB** - Assumes variables have a multinomial distribution. Good for text classification. See `examples/nums.jl` for usage.
 * **GaussianNB** - Assumes variables have a multivariate normal distribution. Good for real-valued data. See `examples/iris.jl` for usage.
 * **KernelNB (depricated)** - Computes kernel density estimates for each class. Good for data with continuous distributions. 
 * **HybridNB** - A hybrid empirical naive Bayes model for a mixuture of continues and discrete features. The continuous feaures are estimated using Kernel Density Estimation (simirarly to KernelBN). Note that the fit/predict methods take a vectors of feature vector `Vector{Vector}` rather than a matrix.


Since `GaussianNB` models multivariate distribution, it's not really a "naive" classifier (i.e. no independence assumption is made), so the name may change in the future.

As a subproduct, this package also provides a `DataStats` type that may be used for incremental calculation of common data statistics such as mean and covariance matrix. See `test/datastatstest.jl` for a usage example.

###Example:

`training_features_continuous = Vector{Vector{Float64}}()`        continuous features as Float64

`push!(training_features_continuous, f_c1, f_c2)`

`training_features_discrete = Vector{Vector{Int}}()`              discrete features (as Int64)

`push!(training_features_discrete, f_d1, f_d2)`

`hybrid_model = HybridNB(labels, length(training_features_continuous), length(training_features_discrete))`

######train the model
`fit(hybrid_model, training_features_continuous, training_features_discrete, labels)`

######predict the classification for new events (points): features_c, features_d
`y = predict(hybrid_model, features_c, features_d)`


**If all the features are continuous a feature matrix is supported:**

`X_tarin = randn(3,400)`

`X_classify = randn(3,10)`

`hybrid_model = HybridNB(labels, size(X, 1)) # the number of discrete features is 0 so it's not needed`

`fit(hybrid_model, X_tarin, labels)`

`y = predict(hybrid_model, X_classify)`
