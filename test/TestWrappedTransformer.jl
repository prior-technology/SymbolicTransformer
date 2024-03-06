using Transformers.HuggingFace
using Transformers.TextEncoders
using SymbolicTransformer.WrappedTransformer

encoder, model = hgf"EleutherAI/pythia-14m"

@testset "embed" begin
    T = prompt(model, encoder, "Hello, world!")    
    @test T.prompt == "Hello, world!"
    
    tokens = encode(encoder, " word").token
    residuals = embed(T, " word")
    r=residuals[1]
    @test r.label == " word"
    @test typeof(r.vector) == Vector{Float32}
    @test r.expression == :(embed(" word"))
end
@testset "inference" begin
    T = prompt(model, encoder, "1 2 3")    
  
    residuals = embed(T, " 4")
    r=residuals[1]
    y = T * r

    @test typeof(y) == HGFResidual
end

