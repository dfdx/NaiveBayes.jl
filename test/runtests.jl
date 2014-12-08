
using NaiveBayes

m = MultinomialNB([:a, :b, :c], 5)
X = [1 2 5 2;
     5 3 -2 1;
     0 2 1 11;
     6 -1 3 3;
     5 7 7 1]
y = [:a, :b, :a, :c]

fit(m, X, y)
@assert predict(m, X) == y

println("All OK.")
