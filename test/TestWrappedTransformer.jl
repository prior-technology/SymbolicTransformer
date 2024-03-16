using Transformers.HuggingFace
using Transformers.TextEncoders
using SymbolicTransformer.WrappedTransformer

encoder = hgf"EleutherAI/pythia-14m:tokenizer"
model = hgf"EleutherAI/pythia-14m:forcausallm"

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

@testset "unembed" begin
    #given
    T = prompt(model, encoder, "Hello,")    
    tokens = encode(encoder, " world").token
    tokenid = first(reinterpret(Int32, tokens))    
    output_vector = T.unembed.layer.embed.embeddings[:,tokenids[1]]
    
    #when
    residuals = unembed(T, " world")
    r=first(residuals)

    #then
    @test r.vector == output_vector
    @test r.label == "< world|"
    @test typeof(r.vector) == Vector{Float32}
    @test r.expression == :(unembed(" world"))
end






@testset "inference" begin
    T = prompt(model, encoder, "1 2 3")    
  
    residuals = embed(T, " 4")
    r=residuals[1]
    y = T * r

    logits = logits(T,y)
    tokenid = argmax(logits)
    token = decode(encoder, tokenid)
    @test token == " 5"
    @test typeof(y) == HGFResidual
end

