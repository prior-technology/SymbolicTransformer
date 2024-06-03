export expand

"Replace a prediction with the contribution to the prediction from each block of the transformer"
function expand(T::PromptedTransformer, prediction::Prediction, input::Residual)
    
    (ln, blocks) = extract_blocks(T)
    
    blockOutputs = [input]
    for (i,block) in enumerate(blocks)
        blockOutput = block * sum(blockOutputs)
        expression = :($(block.expression) * sum(blockOutputs[range(1,$i)]))
        label = """B$i("$(input.label)")"""
        blockOutput = Residual(blockOutput.vector, expression, label)
        push!(blockOutputs, blockOutput)
    end
    
    #<x, LN (a + b)> =  \frac{\sqrt{N}}{\sqrt{|c(a+b)|^2 + N \epsilon} } (<x,c(a)> + <x, c(b)>) 
    N = length(input.vector)
    (lhs, rhs) = prediction_terms(prediction)
    scale = sqrt(N) / sqrt(norm_square(center(sum(blockOutputs))) + N * ln.ϵ)
    centeredBlockOutputs = map(residual -> center(residual), blockOutputs)
    transformedBlockOutputs = map(residual -> gain(ln, residual), centeredBlockOutputs)
    push!(transformedBlockOutputs, Residual(ln.β, :(β), "β"))
    return [

        PredictionTerm(
            prediction.unembed, 
            residual, 
            scale, 
            prediction.normalization_constant, 
            prediction.max_logit, 
            logit(prediction),
            if (i==1) 
                :($lhs ⋅ center($(input.expression)))
            elseif (i==length(transformedBlockOutputs))
                :($lhs ⋅ T.ln.β)
            else
                :($lhs ⋅ expand(T, $rhs)[$i])
            end
        ) 
        for (i,residual) in enumerate(transformedBlockOutputs)
    ]
end
