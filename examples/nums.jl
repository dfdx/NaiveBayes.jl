
using NaiveBayes

X = [1 1 0 2 1;
     0 0 3 1 0;
     1 0 1 0 2]

y = [:a, :b, :b, :a, :a]

m = MultinomialNB(unique(y), 3)
fit(m, X, y)


Xtest = [0 4 1;
         2 2 0;
         1 1 1]

predict(m, Xtest)
