using SymbolicTransformer

@testset "VectorTransformer.jl" begin
    config = Pythia70ModelConfig()
    #check that getter is a function 
    @test @isdefined config
end