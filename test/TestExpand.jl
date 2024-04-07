using SymbolicTransformer
using Test
using Transformers.HuggingFace
using Transformers.TextEncoders
using SymbolicTransformer.WrappedTransformer
using TextEncodeBase
using LinearAlgebra

const encoder = hgf"EleutherAI/pythia-14m:tokenizer"
const model = hgf"EleutherAI/pythia-14m:forcausallm"

function test_expand_residual()
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

function test_extract_blocks()
    #given a PromptedTransformer
    T = prompt(model, encoder, "1, 2, 3, 4")

    #when I extract_blocks
    (ln, prompt_blocks) = extract_blocks(T)

    #then the residuals of the first block should be the same as the prompt
    @test prompt_blocks[1].prompt_residuals == prompt_residuals(T)
    #and ln(sum(prompt_block)) should be the same as the output of the model applied to the prompt residuals
    
    residual_out = residual_in = expected = first(T * embed(T, ","))
     
    for block in prompt_blocks
        residual_out = residual_out + (block * residual_in)        
    end
    result = ln((; hidden_state=first(residual_out).vector))
    @test result.hidden_state ≈ expected.vector
end