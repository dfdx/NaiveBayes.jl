using Random
using LinearAlgebra

kde_names(m::HybridNB) = collect(keys(m.c_kdes[m.classes[1]]))
discrete_names(m::HybridNB) = collect(keys(m.c_discrete[m.classes[1]]))

function compare_models!(m3::HybridNB, m4::HybridNB)
    @test m3.classes == m4.classes
    @test m3.priors == m4.priors
    @test kde_names(m3) == kde_names(m4)
    @test discrete_names(m3) == discrete_names(m4)

    for c in m3.classes
        for (p1, p2) = zip(m3.c_discrete[c], m4.c_discrete[c])
            @test p1.second.pairs == p2.second.pairs
            @test p1.first == p2.first
        end
        for (p1, p2) in zip(m3.c_kdes[c], m4.c_kdes[c])
            @test p1.first == p2.first
            @test p1.second.kde.x == p2.second.kde.x
            @test p1.second.kde.density == p2.second.kde.density
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

        Random.seed!(0)

        # test with names as Symbols
        perm = Random.randperm(N1+N2)
        labels = [ones(Int, N1); zeros(Int, N2)][perm]
        f_c1 = [0.35randn(N1); 3.0 .+ 0.2randn(N2)][perm]
        f_c2 = [-4.0 .+ 0.35randn(N1); -3.0 .+ 0.2randn(N2)][perm]
        f_d = [rand(1:10, N1); rand(12:25, N2)][perm]

        N = AbstractString
        training_c = Dict{N, Vector{Float64}}("c1" => f_c1[1:end-Np],   "c2" => f_c2[1:end-Np])
        predict_c  = Dict{N, Vector{Float64}}("c1" => f_c1[end-Np:end], "c2" => f_c2[end-Np:end])
        training_d = Dict{N, Vector{Int}}("d1" => f_d[1:end-Np])
        predict_d  = Dict{N, Vector{Int}}("d1" => f_d[end-Np:end])

        model = train(HybridNB, training_c, training_d, labels[1:end-Np])
        y_h = predict(model, predict_c, predict_d)
        @test all(y_h .== labels[end-Np:end])

        mktempdir() do dir
            write_model(model, joinpath(dir, "test.h5"))
            m2 = load_model(joinpath(dir, "test.h5"))
            compare_models!(model, m2)
        end


        #testing reading and writing the model file with Symbols
        m3 = HybridNB(y)
        fit(m3, X, y)
        @test all(predict(m3, X) .== y)

        mktempdir() do dir
            write_model(m3, joinpath(dir, "test.h5"))
            m4 = load_model(joinpath(dir, "test.h5"))
            compare_models!(m3, m4)
        end
    end

    @testset "Restructure features" begin
        M = rand(3, 4)
        V = restructure_matrix(M)
        Mp = to_matrix(V)
        @test all(M .== Mp)
    end

    @testset "Multinomial NB - probabistic predictions" begin

        # some word counts in children's books about colours:
        red = [2 0 1 0 1]
        blue = [4 1 2 3 2]
        green = [0 2 0 6 1]
        X = vcat(red, blue, green)
        X_sparse = sparse(X)

        # gender of author:
        y = [:m, :f, :m, :f, :m]

        # Lagrangian smoothing replaces above data with
        # red = [2, 0, 1, 0, 1, 1, 1]'
        # blue = [4, 1, 2, 3, 2, 1, 1]'
        # green = [0, 2, 0, 6, 1, 1, 1]'
        # y = [:m, :f, :m, :f, :m, :m, :f]
        # which gives total class counts of :f => 3, :m => :4

        # working out, by hand, estimates of p(color=red|male), etc,
        # with Lagrangian smoothing:
        red_given_m = 5/16
        blue_given_m = 9/16
        green_given_m = 2/16
        red_given_f = 1/15
        blue_given_f = 5/15
        green_given_f = 9/15

        # let `m(r, b, g)` be Naive Bayes prediction of probablity of
        # class `:m`, given counts `red=r`, `blue=b` and
        # `green=g`. Similar for `f(r, b, g)`:
        m_(red, blue, green) =
            4/7*(red_given_m^red)*(blue_given_m^blue)*(green_given_m^green)
        f_(red, blue, green) =
            3/7*(red_given_f^red)*(blue_given_f^blue)*(green_given_f^green)
        normalizer(red, blue, green) =
            m_(red, blue, green) + f_(red, blue, green)
        m(a...) = m_(a...)/normalizer(a...)
        f(a...) = f_(a...)/normalizer(a...)

        # new data:
        red = [1 1]
        blue = [1 2]
        green = [1 3]
        Xnew = vcat(red, blue, green)
        Xnew_sparse = sparse(Xnew)

        # now get NaiveBayes.jl  predictions:
        model = MultinomialNB([:m, :f], 3, alpha=1)
        fit(model, X, y)
        classes, logprobs = predict_logprobs(model, Xnew)
        @test classes == [:f, :m] # implementation changes might
                                  # change order here?
        probs = exp.(logprobs)
        # NaiveBayes does not normalize probabilities, so:
        col_sums = sum(probs, dims=1)
        probs = probs ./ col_sums
        probs_f = probs[1,:]
        probs_m = probs[2,:]

        # compare with above:
        @test m(Xnew[:,1]...) ≈ probs_m[1]
        @test m(Xnew[:,2]...) ≈ probs_m[2]
        @test f(Xnew[:,1]...) ≈ probs_f[1]
        @test f(Xnew[:,2]...) ≈ probs_f[2]

        # test with sparse
        model_sparse = MultinomialNB([:m, :f], 3, alpha=1)
        fit(model_sparse, X_sparse, y)

        @test m(Xnew_sparse[:,1]...) ≈ probs_m[1]
        @test m(Xnew_sparse[:,2]...) ≈ probs_m[2]
        @test f(Xnew_sparse[:,1]...) ≈ probs_f[1]
        @test f(Xnew_sparse[:,2]...) ≈ probs_f[2]

    end

end
