module WrappedTransformer
using Transformers
using Transformers.TextEncoders
using Transformers.HuggingFace
using SymbolicTransformer

export PromptedTransformer, HGFResidual, prompt, embed, unembed, logits

"Wraps a transformer and encoder with a prompt"
struct PromptedTransformer <: SymbolicTransformer.Operation
    "Huggingface pretrained model"
    model 
    "TextEncoder corresponding with model"
    encoder
    "Embedding layer"
    embed
    "Output layer which maps residual vectors to logits"
    unembed
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
function prompt(causal_lm_model::Transformers.HuggingFace.HGFGPTNeoXForCausalLM,
        encoder,
        utterance)
    model = causal_lm_model.model
    unembed = causal_lm_model.cls
    embed = model.embed
    
    tokens = encode(encoder, utterance).token

    return PromptedTransformer(model, encoder, embed, unembed, utterance, tokens, :(T))
end

"tokenizes the utterance, and returns a Vector of Residuals representing the embedding vectors"
function embed(transformer, utterance)    
    tokens = encode(transformer.encoder, utterance).token
    labels = decode(transformer.encoder,tokens)
    vectors = transformer.embed((; token=tokens))
    expressions = map(x -> :(embed($x)), labels)
    residuals = map(x -> 
        HGFResidual(vectors.hidden_state[:,x],
            expressions[x], 
            labels[x]), 
        1:length(labels))
    return residuals
end



function ket(s::AbstractString)
    return "<" * s * "|"
end

"tokenizes the utterance, and returns a Vector of Residuals which map output residuals to logits"
function unembed(transformer, utterance)    
    tokens = encode(transformer.encoder, utterance).token
    labels =  decode(transformer.encoder,tokens)
    tokenids = reinterpret(Int32, tokens)
    output_vectors = transformer.unembed.layer.embed.embeddings[:,tokenids]
    
    expressions = map(x -> :(unembed($x)), labels)
    residuals = map(x -> 
        HGFResidual(output_vectors[:,x],
            expressions[x], 
            ket(labels[x])), 
        1:length(labels))
    return residuals
end

"applies the model to the token"
function Base.:(*)(T::PromptedTransformer, r:: HGFResidual)
    #To transform a new token at the end of a batch of tokens, we would push! the index of the 
    #new token onto tokens.onehots, which applies a corresponding change to the tokens OneHotArray
    
    #In this case we want to pass in an arbitrary residual vector, so we should bypass the embedding layer
    input = (; token=T.tokens)
    residuals = T.embed(input)
    hidden_state = hcat(residuals.hidden_state, r.vector)
    y = T.model.decoder((; hidden_state=hidden_state))
    return HGFResidual(y.hidden_state, :((T.expression) * (r.expression)), string("T ", r.label))
end

function logits(T::PromptedTransformer,r:: HGFResidual)
    (vector, l) = T.unembed((; hidden_state=r.vector))
    
    return l[:,1]
end
end