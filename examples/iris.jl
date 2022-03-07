
using NaiveBayes
using RDatasets
using StatsBase
using Random


# Example 1
iris = dataset("datasets", "iris")

# observations in columns and variables in rows
X = Matrix(iris[:,1:4])'
p, n = size(X)
# by default species is a PooledDataArray,
y = [species for species in iris[:, 5]]

# how much data use for training
train_frac = 0.9
k = floor(Int, train_frac * n)
idxs = randperm(n)
train_idxs = idxs[1:k]
test_idxs = idxs[k+1:end]

model = GaussianNB(unique(y), p)
fit(model, X[:, train_idxs], y[train_idxs])

accuracy = count(!iszero, predict(model, X[:,test_idxs]) .== y[test_idxs]) / count(!iszero, test_idxs)
println("Accuracy: $accuracy")

# Example 2
# 3 classes and 100 random data samples with 5 variables.
n_obs = 100
m = GaussianNB([:a, :b, :c], 5)
X = randn(5, n_obs)
y = sample([:a, :b, :c], n_obs)
fit(m, X, y)
accuracy = sum(predict(m, X) .== y) / n_obs
println("Accuracy: $accuracy")
