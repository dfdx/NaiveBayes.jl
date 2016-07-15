NaiveBayes.jl
=============

[![Build Status](https://travis-ci.org/dfdx/NaiveBayes.jl.svg)](https://travis-ci.org/dfdx/NaiveBayes.jl)

Naive Bayes classifier. Currently 3 types of NB are supported:

 * **MultinomialNB** - Assumes variables have a multinomial distribution. Good for text classification. See `examples/nums.jl` for usage.
 * **GaussianNB** - Assumes variables have a multivariate normal distribution. Good for real-valued data. See `examples/iris.jl` for usage.
 * **HybridNB** - A hybrid empirical naive Bayes model for a mixture of continuous and discrete features. The continuous features are estimated using Kernel Density Estimation.
*Note*: fit/predict methods take `Vector{Vector}` rather than a `Matrix`. Also, discrete features must be integers while continuous features must be floats.


Since `GaussianNB` models multivariate distribution, it's not really a "naive" classifier (i.e. no independence assumption is made), so the name may change in the future.

As a subproduct, this package also provides a `DataStats` type that may be used for incremental calculation of common data statistics such as mean and covariance matrix. See `test/datastatstest.jl` for a usage example.

###Examples:
1. Continuous and discrete features as `Vector{Vector}}`
    ```julia

    f_c1 = randn(10)
    f_c2 = randn(10)
    training_features_continuous = Vector{Vector{Float64}}() #continuous features as Float64
    push!(training_features_continuous, f_c1, f_c2)
    training_features_discrete = Vector{Vector{Int}}() #discrete features as Int64
    f_c1 = rand(1:5, 10)
    f_c2 = randn(3:7, 10)
    push!(training_features_discrete, f_d1, f_d2)

    # initialize the naive bayes model
    hybrid_model = HybridNB(labels, length(training_features_continuous), length(training_features_discrete))

    # train the model
    fit(hybrid_model, training_features_continuous, training_features_discrete, labels)

    # predict the classification for new events (points): features_c, features_d
    y = predict(hybrid_model, features_c, features_d)
    ```

2. Continuous features only as a `Matrix`
    ```julia
    X_tarin = randn(3,400);
    X_classify = randn(3,10)

    hybrid_model = HybridNB(labels, size(X, 1)) # the number of discrete features is 0 so it's not needed
    fit(hybrid_model, X_tarin, labels)
    y = predict(hybrid_model, X_classify)
    ```
3. Continuous and discrete features as a `Matrix{Float}`
    ```julia
        #X is a matrix of features
    # the first 3 rows are continuous
    training_features_continuous = from_matrix(X[1:3, :])
    # the last 2 rows are discrete and must be integers
    training_features_discrete = map(Int, from_matrix(X[4:5, :]))

    hybrid_model = HybridNB(labels, length(training_features_continuous), length(training_features_discrete))

    # train the model
    fit(hybrid_model, training_features_continuous, training_features_discrete, labels)

    # predict the classification for new events (points): features_c, features_d
    y = predict(hybrid_model, features_c, features_d)
    ```


### Write/Load models to files

It is useful to train a model once and than use it for prediction many times later. For example, train your clussifier on a local machine and than use it on a cluster to classify points in parallel.

There is support for writing `HybridNB` models to HDF5 files via the methods `write_model` and `load_model`. This is useful for interacting with other programs/languages. For Julia to Julia it is easy to use **JLD.jl**.
