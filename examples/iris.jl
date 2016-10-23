
using NaiveBayes
using RDatasets
using StatsBase

# Example 1
iris = dataset("datasets", "iris")

# observations in columns and variables in rows
X = convert(Array, iris[:, 1:4])'
p, n = size(X)
# by default species is a PooledDataArray,
y = [species for species in iris[:, 5]]

# how much data use for training
train_frac = 0.9
k = floor(Int, train_frac * n)
idxs = randperm(n)
train = idxs[1:k]
test = idxs[k+1:end]

model = GaussianNB(unique(y), p)
fit(model, X[:, train], y[train])

accuracy = countnz(predict(model, X[:,test]) .== y[test]) / countnz(test)
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
