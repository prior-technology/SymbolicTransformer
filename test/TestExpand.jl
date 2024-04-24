using SymbolicTransformer
using Test
using Transformers.HuggingFace
using Transformers.TextEncoders
using SymbolicTransformer.WrappedTransformer
using TextEncodeBase
using LinearAlgebra


function test_expand_residual()
    (model, encoder) = TestData.get_both()

    #given an output residual from applying a prompted transformer to a residual
    T = prompt(model, encoder, "1, 2, 3, 4")
    input = embed(T, ",")
    output = T * input
    
    #when I expand the residual
    expanded_residual = expand(T, output)

    #then the expanded residual should include several terms which combined result in the original residual
    @test length(expanded_residual) == 6 # 6 blocks in the transformer
    @test sum(map(r -> r.vector, expanded_residual)) ≈ output.vector
end

function test_expand_expression()
    (model, encoder) = TestData.get_both()

    #given an expression which applies a prompted transformer to a residual
    T = prompt(model, encoder, "1, 2, 3, 4")
    input = embed(T, ",")
    expression = :(T * input)
    
    #when I expand the expression
    expanded_expression = expand(:T, expression)

    #then the expanded expression should include several terms which combined result in the original residual
    @test expanded_expression == :((LayerNorm ∘ Block[6] ∘ Block[5] ∘ Block[4] ∘ Block[3] ∘ Block[2] ∘ Block[1]) * input)

    #or to put it another way
    @test expanded_expression == :(LayerNorm( x + BlockOutput[1] + BlockOutput[2] + BlockOutput[3] + BlockOutput[4] + BlockOutput[5] + BlockOutput[6] ) )

end
function test_expand_prediction()
    (model, encoder) = TestData.get_both()

    #given a prediction
    T = prompt(model, encoder, "1, 2, 3, 4")
    input = embed(T, ",")
    residual = first(T * input)
    prediction = first(predict(T, residual))

    #when I expand the prediction
    expanded_prediction = expand(T, prediction)

    #then the expanded prediction should include several terms which combined result in the original prediction
    @test length(expanded_prediction) == 8 # 6 blocks in the transformer + input residual + bias
    @test sum(map(p -> probability(p), expanded_prediction)) ≈ probability(prediction)

end
function test_extract_blocks()
    (model, encoder) = TestData.get_both()

    #given a PromptedTransformer
    T = prompt(model, encoder, "1, 2, 3, 4")

    #when I extract_blocks
    (ln, prompt_blocks) = extract_blocks(T)

    #then the residuals of the first block should be the same as the prompt
    @test prompt_blocks[1].prompt_residuals == prompt_residuals(T)
    #and ln(sum(prompt_block)) should be the same as the output of the model applied to the prompt residuals
    
    this_block_residual = total_residual = first(embed(T, ","))
    
    for block in prompt_blocks
        this_block_residual = block * total_residual 
        total_residual  = total_residual + this_block_residual
    end
    result = ln(total_residual.vector)
 
    expected = first(T * embed(T, ","))
    
    @test result ≈ expected.vector
    
end

#test_extract_blocks()
test_expand_prediction()
