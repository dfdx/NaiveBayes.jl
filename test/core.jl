function compare_models!(m3::HybridNB, m4::HybridNB)
    @test m3.classes == m4.classes
    @test m3.priors == m4.priors
    @test m3.kde_names == m4.kde_names
    @test m3.discrete_names == m4.discrete_names

    for c in m3.classes
        for (p1, p2) = zip(m3.c_discrete[c], m4.c_discrete[c])
            @test p1.pairs == p2.pairs
        end
        for (k1, k2) = zip(m3.c_kdes[c], m4.c_kdes[c])
            @test k1.kde.x == k2.kde.x
            @test k1.kde.density == k2.kde.density
        end
    end

end


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

    @testset "Hybrid NB" begin

        N1 = 100000
        N2 = 160000
        Np = 1000

        srand(0)

        # test with names as Symbols
        perm = randperm(N1+N2)
        labels = [ones(Int, N1); zeros(Int, N2)][perm]
        f_c1 = [0.35randn(N1); 3.0 + 0.2randn(N2)][perm]
        f_c2 = [-4.0 + 0.35randn(N1); -3.0 + 0.2randn(N2)][perm]
        f_d = [rand(1:10, N1); rand(12:25, N2)][perm]

        training_c = Vector{Vector{Float64}}()
        predict_c = Vector{Vector{Float64}}()
        push!(training_c, f_c1[1:end-Np], f_c2[1:end-Np])
        push!(predict_c, f_c1[end-Np:end], f_c2[end-Np:end])
        names_c = [:c1, :c2]

        training_d = Vector{Vector{Int}}()
        predict_d = Vector{Vector{Int}}()
        push!(training_d, f_d[1:end-Np])
        push!(predict_d, f_d[end-Np:end])
        names_d = [:d1]

        # Test constructor with the number of features
        m = HybridNB(labels[1:end-Np], length(names_c), length(names_d))
        fit(m, training_c, training_d, labels[1:end-Np])
        y_h = predict(m, predict_c, predict_d)
        @test all(y_h .== labels[end-Np:end])

        # Test constructor with given feature names
        model = HybridNB(labels[1:end-Np], names_c, names_d)
        fit(model, training_c, training_d, labels[1:end-Np])
        y_h = predict(model, predict_c, predict_d)
        @test all(y_h .== labels[end-Np:end])

        mkdir("tmp")
        write_model(model, "tmp/test.h5")
        m2 = load_model("tmp/test.h5")
        rm("tmp", recursive=true)
        compare_models!(model, m2)


        # Test reading and writing the model file with Strings
        m3 = HybridNB(y, ["c1", "c2"])
        fit(m3, X, y)
        @test predict(m3, X) == y

        mkdir("tmp")
        write_model(m3, "tmp/test.h5")
        m4 = load_model("tmp/test.h5")
        rm("tmp", recursive=true)
        compare_models!(m3, m4)
    end

    @testset "Restructure features" begin
        M = rand(3, 4)
        V = restructure_matrix(M)
        Mp = to_matrix(V)
        @test all(M .== Mp)
    end
end
