NaiveBayes.jl
=============

[![Build Status](https://travis-ci.org/dfdx/NaiveBayes.jl.svg)](https://travis-ci.org/dfdx/NaiveBayes.jl)

Naive Bayes classifier. Currently 3 types of NB are supported:

 * **MultinomialNB** - Assumes variables have a multinomial distribution. Good for text classification. See `examples/nums.jl` for usage.
 * **GaussianNB** - Assumes variables have a multivariate normal distribution. Good for real-valued data. See `examples/iris.jl` for usage.
 * **KernelNB** - Computes kernel density estimates for each class. Good for data with continuous distributions.

Since `GaussianNB` models multivariate distribution, it's not really a "naive" classifier (i.e. no independence assumption is made), so the name may change in the future.

As a subproduct, this package also provides a `DataStats` type that may be used for incremental calculation of common data statistics such as mean and covariance matrix. See `test/datastatstest.jl` for a usage example.
