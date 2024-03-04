module WrappedTransformer
using Transformers
using Transformers.TextEncoders
using Transformers.HuggingFace
using SymbolicTransformer

"Wraps a transformer and encoder with a prompt"
struct PromptedTransformer <: SymbolicTransformer.Operation
    "Huggingface pretrained model"
    model 
    "TextEncoder corresponding with model"
    encoder 
    "Original string of the prompt"
    prompt :: AbstractString
    "result of Transformers.TextEncoders.encode - nvocab x ntokens OneHotArray"
    tokens
    "Simple expression representing this Transformer"
    expression 
end

"Represents a vector in the transformer's residual space"
struct HGFResidual <:  SymbolicTransformer.Residual
    "Top level transformer for this residual"
    transformer 
    "vector in the residual space"
    vector 
    "Expression showing the source of this residual"
    expression
    "Label for printing"
    label
end


"tokenizes the utterance, and returns an operation"
function prompt(model,
        encoder,
        utterance)
    
    tokens = encode(encoder, utterance).token

    return PromptedTransformer(model, encoder, utterance, tokens, :(T))
end

"returns a Residual representing embedding a single token"
function residual(transformer, token)
    label = decode(transformer.encoder,token)
    vector = transformer.model.embed((; token=token))
    expression = :(embed($label))
    return HGFResidual( transformer, vector, expression, label)
end

# function embed(PromptedTransformer, utterance)
#     "tokenizes the utterance, and returns a Vector of Residuals"
#     tokens = encode(encoder, utterance).token
#     #embeddings = model.embed(tokens)
#     return map(token -> residual(model, encoder, token), tokens.onehots)

# end
# function *(T::PromptedTransformer, r:: SymbolicTransformer.Residual)
#     "applies the model to the token"
#     y = T.model(x)
#     return HGFResidual(T.model, T.encoder, y, :((T.expression) * (r.expression)), string(T, " ", r.label))
# end

end