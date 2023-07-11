using SymbolicTransformer.SymbolCreator

@testset "SymbolCreator.jl" begin
    cache = Dict("test" => "value")
    getter = SymbolCreator.TransformerLensCacheParser(cache)
    #check that getter is a function 
    @test @isdefined getter
end