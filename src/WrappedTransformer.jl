using Transformers
using Transformers.TextEncoders
using Transformers.HuggingFace

module WrappedTransformer
"Wraps model from Transformer.jl"

struct PromptedTransformer{M <: Transformers.HuggingFace.HGFPreTrainedModel, E <: Transformers.TextEncoders.AbstractTransformerTextEncoder} <: Operation
    model :: M
    encoder :: E
    prompt :: AbstractString
    expression :: Expr      
end
struct HGFResidual{M <: Transformers.HuggingFace.HGFPreTrainedModel, E <: Transformers.TextEncoders.AbstractTransformerTextEncoder} <: Residual
    model :: M
    encoder :: E
    vector :: AbstractVector
    expression :: Expr
    label :: AbstractString
end


function prompt(model :: Transformers.HuggingFace.HGFPreTrainedModel,encoder,utterance)
    "tokenizes the utterance, and returns an operation"
    tokens = tokenize(encoder, utterance)
    return PromptedTransformer(model, encoder, utterance, tokens)
end

function residual(model:: Transformers.HuggingFace.HGFPreTrainedModel, 
        encoder :: Transformers.TextEncoders.AbstractTransformerTextEncoder,
        token)
    "returns a Residual"
    label = decode(encoder,token)
    vector = model.embed((; token=token))
    expression = :(embed($label))
    return HGFResidual( model, encoder, vector, expression, label)
end
function embed(model, encoder, utterance)
    "tokenizes the utterance, and returns a Vector of Residuals"
    tokens = encode(encoder, utterance).token
    #embeddings = model.embed(tokens)
    return map(token -> residual(model, encoder, token), tokens.onehots)

end
function *(T::PromptedTransformer, r::Residual)
    "applies the model to the token"
    y = T.model(x)
    return HGFResidual(T.model, T.encoder, y, :((T.expression) * (r.expression)), string(T, " ", r.label))
end

end