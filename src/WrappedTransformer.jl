using Transformers
using Transformers.TextEncoders

module WrappedTransformer
"Wraps model from Transformer.jl"

#PythiaTransformer inherits Operation
struct PythiaTransformer <: Operation
    forward
    expression    
    textencoder
    model
end
struct Token
    id
    text
    position
    vector
end
function transform(utterance)
    "This method tokenizes the utterance, and returns an operation"
    textencoder, model = hgf"EleutherAI/pythia-70m-deduped"
    tokens = tokenize(textencoder, utterance)
    apply_transformer = function(residual :: Residual)
        tokens = append!(tokens, residual.vector)
        
        for block in model.blocks
            residuals = residuals + apply_transformer_block(model.config, block, residuals)
        end
    end
    return PythiaTransformer(apply_transformer, Expr(:block, tokens), textencoder, model)
end


end