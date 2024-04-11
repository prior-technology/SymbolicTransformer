using SymbolicTransformer
using Test
using Transformers.HuggingFace
using Transformers.TextEncoders
using SymbolicTransformer.WrappedTransformer
using TextEncodeBase
using LinearAlgebra


function test_embed()
    (model, encoder) = TestData.get_both()
    
    T = prompt(model, encoder, "Hello, world!")    
    @test T.prompt == "Hello, world!"    
    residuals = embed(T, " word")
    r=residuals[1]
    @test r.label == " word"
    @test typeof(r.vector) == Vector{Float32}
    @test r.expression == :(embed(" word"))
end

function test_unembed()
    (model, encoder) = TestData.get_both()

    #given
    T = prompt(model, encoder, "Hello,")    
    tokens = encode(encoder, " world").token
    token_ids = first(reinterpret(Int32, tokens))    
    output_vector = T.unembed_layer.layer.embed.embeddings[:,token_ids[1]]
    
    #when
    residuals = unembed(T, " world")
    r=first(residuals)

    #then
    @test r.vector == adjoint(output_vector)
    @test r.label == " world"
    @test r.expression == :(unembed(" world"))

end

function test_logits()
    (model, encoder) = TestData.get_both()

    #given an output residual which matches a specific vector of the unembedding layer
    T = prompt(model, encoder, "Hello")
    residuals = unembed(T, "Hello")
    r=transpose(first(residuals))

    #When I calculate the logits for that residual
    predictions = predict(T,r)

    #Then the logit for that token should be >> than the next closest    
    tokenid = argmax(map(p -> p.logit, predictions))
    @test predictions[tokenid].label == "Hello"
end

function test_inference()
    (model, encoder) = TestData.get_both()

    #given a transformer prompted with a sequence of numbers
    T = prompt(model, encoder, "1, 2, 3, 4")     

    #when the transformer operates on a residual of a token which continues the sequence
    residuals = embed(T, ",")
    r=residuals[1]
    y = T * r
    @test typeof(y) == HGFResidual
    predictions = predict(T,y)
    p = first(predictions)

    #then the transformer should predict the next number in the sequence    
    @test p.label == " 5"
    @test p.probability > 0.25
    @test p.expression == :(unembed(" 5") ⋅ (T * embed(",")))

    #and the logit should match the equivalent when using transformers.jl directly
    tjlInput = encode(encoder, "1, 2, 3, 4,")
    tjlOutput = model(tjlInput)
    @test p.logit ≈ tjlOutput.logit[p.token_id,end,1] #token_id from vocab, end of sequence, batch 1

    #and the logit should match the result of it's own expression
    inner_product = first((unembed(" 5") ⋅ (T * embed(","))))
    @test p.logit ≈ inner_product.vector[1]
end

@testset "embed" test_embed()
@testset "unembed" test_unembed()
@testset "logits" test_logits()
@testset "inference" test_inference()
