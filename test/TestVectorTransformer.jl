using SymbolicTransformer

@testset "VectorTransformer.jl" begin
    config = Pythia70ModelConfig()
    #check that getter is a function 
    @test @isdefined config
end

@testset "inverse_frequencies" begin
    freqs = SymbolicTransformer.inverse_frequencies(100,8)
    @test length(freqs) == 5
    @test (1.0, 0.1, 0.01) == (freqs[1], freqs[3], freqs[5])
end

@testset "rotate_half" begin
    m = [1 2 3 4 5; 10 20 30 40 50]
    r = SymbolicTransformer.rotate_half(m)
    @test r == [-3 -4 -5 1 2; -30 -40 -50 10 20]
end