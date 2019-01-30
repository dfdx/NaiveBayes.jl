NaiveBayes.jl
=============

[![Build Status](https://travis-ci.org/dfdx/NaiveBayes.jl.svg)](https://travis-ci.org/dfdx/NaiveBayes.jl)
[![codecov.io](http://codecov.io/github/dfdx/NaiveBayes.jl/coverage.svg)](http://codecov.io/github/dfdx/NaiveBayes.jl)

Naive Bayes classifier. Currently 3 types of NB are supported:

 * **MultinomialNB** - Assumes variables have a multinomial distribution. Good for text classification. See `examples/nums.jl` for usage.
 * **GaussianNB** - Assumes variables have a multivariate normal distribution. Good for real-valued data. See `examples/iris.jl` for usage.
 * **HybridNB** - A hybrid empirical naive Bayes model for a mixture of continuous and discrete features. The continuous features are estimated using Kernel Density Estimation.
*Note*: fit/predict methods take `Dict{Symbol/AstractString, Vector}` rather than a `Matrix`. Also, discrete features must be integers while continuous features must be floats. If all features are continuous `Matrix` input is supported.


Since `GaussianNB` models multivariate distribution, it's not really a "naive" classifier (i.e. no independence assumption is made), so the name may change in the future.

As a subproduct, this package also provides a `DataStats` type that may be used for incremental calculation of common data statistics such as mean and covariance matrix. See `test/datastatstest.jl` for a usage example.

### Examples:
1. Continuous and discrete features as `Dict{Symbol, Vector}}`

    ```julia
    f_c1 = randn(10)
    f_c2 = randn(10)
    f_d1 = rand(1:5, 10)
    f_d2 = randn(3:7, 10)
    training_features_continuous = Dict{Symbol, Vector{Float64}}(:c1=>f_c1, :c2=>f_c2)
    training_features_discrete   = Dict{Symbol, Vector{Int}}(:d1=>f_d1, :d2=>f_d2) #discrete features as Int64

    hybrid_model = HybridNB(labels)

    # train the model
    fit(hybrid_model, training_features_continuous, training_features_discrete, labels)
    # predict the classification for new events (points): features_c, features_d
    y = predict(hybrid_model, features_c, features_d)
    ```
    Alternatively one can skip declaring the model and train it directly:
    ```julia
    model = train(HybridNB, training_features_continuous, training_features_discrete, labels)
    y = predict(hybrid_model, features_c, features_d)
    ```

2. Continuous features only as a `Matrix`
    ```julia
    X_train = randn(3,400);
    X_classify = randn(3,10)

    hybrid_model = HybridNB(labels) # the number of discrete features is 0 so it's not needed
    fit(hybrid_model, X_train, labels)
    y = predict(hybrid_model, X_classify)
    ```
3. Continuous and discrete features as a `Matrix{Float}`
    ```julia
    #X is a matrix of features
    # the first 3 rows are continuous
    training_features_continuous = restructure_matrix(X[1:3, :])
    # the last 2 rows are discrete and must be integers
    training_features_discrete = map(Int, restructure_matrix(X[4:5, :]))
    # train the model
    hybrid_model = train(HybridNB, training_features_continuous, training_features_discrete, labels)

    # predict the classification for new events (points): features_c, features_d
    y = predict(hybrid_model, features_c, features_d)
    ```


### Write/Load models to files

It is useful to train a model once and then use it for prediction many times later. For example, train your classifier on a local machine and then use it on a cluster to classify points in parallel.

There is support for writing `HybridNB` models to HDF5 files via the methods `write_model` and `load_model`. This is useful for interacting with other programs/languages. If the model file is going to be read only in Julia it is easier to use **JLD.jl** for saving and loading the file.
