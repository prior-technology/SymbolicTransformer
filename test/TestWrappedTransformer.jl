using Transformers.HuggingFace
using Transformers.TextEncoders
using SymbolicTransformer.WrappedTransformer

const encoder = hgf"EleutherAI/pythia-14m:tokenizer"
const model = hgf"EleutherAI/pythia-14m:forcausallm"


function test_embed()
    T = prompt(model, encoder, "Hello, world!")    
    @test T.prompt == "Hello, world!"
    
    tokens = encode(encoder, " word").token
    residuals = embed(T, " word")
    r=residuals[1]
    @test r.label == " word"
    @test typeof(r.vector) == Vector{Float32}
    @test r.expression == :(embed(" word"))
end

function test_unembed()
    #given
    T = prompt(model, encoder, "Hello,")    
    tokens = encode(encoder, " world").token
    token_ids = first(reinterpret(Int32, tokens))    
    output_vector = T.unembed.layer.embed.embeddings[:,token_ids[1]]
    
    #when
    residuals = unembed(T, " world")
    r=first(residuals)

    #then
    @test r.vector == output_vector
    @test r.label == "< world|"
    @test typeof(r.vector) == Vector{Float32}
    @test r.expression == :(unembed(" world"))

end

function test_logits()
    #given an output residual which matches a specific vector of the unembedding layer
    T = prompt(model, encoder, "Hello")
    residuals = unembed(T, " world")
    r=first(residuals)

    #When I calculate the logits for that residual
    (expressions, logits) = logits(T,y)

    #Then the logit for that token should be close to 1

end

function test_inference()
    T = prompt(model, encoder, "1 2 3")    

    residuals = embed(T, " 4")
    r=residuals[1]
    y = T * r

    (expressions, logits) = logits(T,y)
    expression_id = argmax(logits)
    expression = expressions(argmax(logits))
    
    @test expression == " 5"
    @test typeof(y) == HGFResidual
end

@testset "embed" test_embed()
@testset "unembed" test_unembed()
#@testset "inference" test_inference()
