using Transformers.HuggingFace
using Transformers.TextEncoders
using SymbolicTransformer

encoder, model = hgf"EleutherAI/pythia-14m"

@testset "residual" begin
    T = SymbolicTransformer.WrappedTransformer.prompt(model, encoder, "Hello, world!")    
    @test T.prompt == "Hello, world!"
    
    tokens = encode(encoder, " word").token
    r = SymbolicTransformer.WrappedTransformer.residual(T, tokens)
    @test r.label == " word"
    @test typeof(r.vector) == Matrix{Float32}
    @test r.expression == :(embed(" word"))
end