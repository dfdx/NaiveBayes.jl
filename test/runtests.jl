
using NaiveBayes

if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

include("core.jl")


m = MultinomialNB([:a, :b, :c], 5)
X1 = [1 2 5 2;
      5 3 -2 1;
      0 2 1 11;
      6 -1 3 3;
      5 7 7 1]

y1 = [:a, :b, :a, :c]
fit(m, X1, y1)
      
@test predict(m, X1) == y1
