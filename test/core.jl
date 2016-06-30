@testset "Core Functions" begin
    # 6 data samples with 2 variables belonging to 2 classes
    X = [-1.0 -2.0 -3.0 1.0 2.0 3.0;
         -1.0  -1.0 -2.0 1.0 1.0 2.0]
    y = [1, 1, 1, 2, 2, 2]

    @testset "Multinomial NB" begin
        m = MultinomialNB([:a, :b, :c], 5)
        X1 = [1 2 5 2;
             5 3 -2 1;
             0 2 1 11;
             6 -1 3 3;
             5 7 7 1]
        y1 = [:a, :b, :a, :c]
        fit(m, X1, y1)
        @test predict(m, X1) == y1
    end

    @testset "Gaussian NB" begin
        m = GaussianNB(unique(y), 2)
        fit(m, X, y)
        @test predict(m, X) == y
    end

    @testset "KDE NB" begin
        m = KernelNB(y, 2)
        fit(m, X, y)
        @test predict(m, X) == y
    end
end
