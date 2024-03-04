module WrappedTransformer
using Transformers
using Transformers.TextEncoders
using Transformers.HuggingFace
using SymbolicTransformer

"Wraps a transformer and encoder with a prompt"
struct PromptedTransformer{M <: Transformers.HuggingFace.HGFPreTrainedModel, E <: Transformers.TextEncoders.AbstractTransformerTextEncoder} <: SymbolicTransformer.Operation
    "Huggingface pretrained model"
    model :: M
    "TextEncoder corresponding with model"
    encoder :: E
    "Original string of the prompt"
    prompt :: AbstractString
    "result of Transformers.TextEncoders.encode - nvocab x ntokens OneHotArray"
    tokens :: OneHotArray
    "Simple expression representing this Transformer"
    expression :: Expr      
end

"Represents a vector in the transformer's residual space"
struct HGFResidual{M , E} <:  SymbolicTransformer.Residual
    "Top level transformer for this residual"
    transformer :: PromptedTransformer
    "vector in the residual space"
    vector :: AbstractVector
    "Expression showing the source of this residual"
    expression :: Expr
    "Label for printing"
    label :: AbstractString
end


"tokenizes the utterance, and returns an operation"
function prompt(model,
        encoder,
        utterance)
    
    tokens = encode(encoder, utterance).token
    return PromptedTransformer(model, encoder, utterance, tokens, :(T))
end

# function residual(model:: Transformers.HuggingFace.HGFPreTrainedModel, 
#         encoder :: Transformers.TextEncoders.AbstractTransformerTextEncoder,
#         token)
#     "returns a Residual"
#     label = decode(encoder,token)
#     vector = model.embed((; token=token))
#     expression = :(embed($label))
#     return HGFResidual( model, encoder, vector, expression, label)
# end
# function embed(model, encoder, utterance)
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