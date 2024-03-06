module WrappedTransformer
using Transformers
using Transformers.TextEncoders
using Transformers.HuggingFace
using SymbolicTransformer

export PromptedTransformer, HGFResidual, prompt, embed

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

"tokenizes the utterance, and returns a Vector of Residuals representing the embedding vectors"
function embed(transformer, utterance)    
    tokens = encode(transformer.encoder, utterance).token
    labels = decode(transformer.encoder,tokens)
    vectors = transformer.model.embed((; token=tokens))
    expressions = map(x -> :(embed($x)), labels)
    residuals = map(x -> 
        HGFResidual(vectors.hidden_state[:,x],
            expressions[x], 
            labels[x]), 
        1:length(labels))
    return residuals
end

"applies the model to the token"
function Base.:(*)(T::PromptedTransformer, r:: HGFResidual)
    #To transform a new token at the end of a batch of tokens, we would push! the index of the 
    #new token onto tokens.onehots, which applies a corresponding change to the tokens OneHotArray
    
    #In this case we want to pass in an arbitrary residual vector, so we should bypass the embedding layer
    input = (; token=T.tokens)
    residuals = T.model.embed(input)
    hidden_state = hcat(residuals.hidden_state, r.vector)
    y = T.model.decoder((; hidden_state=hidden_state))
    return HGFResidual(y.hidden_state, :((T.expression) * (r.expression)), string("T ", r.label))
end

end