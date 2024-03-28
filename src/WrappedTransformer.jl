module WrappedTransformer
using Transformers
using Transformers.TextEncoders
using Transformers.HuggingFace
using SymbolicTransformer
using LinearAlgebra
import Base.show

export PromptedTransformer, HGFResidual, prompt, embed, unembed, predict, dot

"Wraps a transformer and encoder with a prompt"
struct PromptedTransformer <: SymbolicTransformer.Operation
    "Huggingface pretrained model"
    model 
    "TextEncoder corresponding with model"
    encoder
    "Embedding layer"
    embed_layer
    "Output layer which maps residual vectors to logits"
    unembed_layer
    "Original string of the prompt"
    prompt :: AbstractString
    "result of Transformers.TextEncoders.encode - nvocab x ntokens OneHotArray"
    tokens
    "Simple expression representing this Transformer"
    expression 
end

global current_transformer::PromptedTransformer

function show(io::IO, T::PromptedTransformer)
    show(io, MIME("text/plain"), T)
end
function show(io::IO, ::MIME"text/plain", T::PromptedTransformer)
    
    if (get(io, :compact, false) == true)
        print(io, "PromptedTransformer(\"$(T.prompt)\")")
    else
        #Display the model type, encoder type and prompt
        #typeof(T.model) is quite complex, simplify it
        model_type = split(string(typeof(T.model)), "{")[1]
        encoder_type = split(string(typeof(T.encoder)), "{")[1]
        print(io, "PromptedTransformer($model_type, $encoder_type, \"$(T.prompt)\")")        
    end
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
function show(io::IO, ::MIME"text/plain", r::HGFResidual)
    if (get(io, :compact, false) == true)
        print(io, r.expression)
    else
        print(io, "HGFResidual(\"$(r.label)\", $(r.expression))")
    end
end
struct Prediction <: SymbolicTransformer.Prediction
    token_id
    logit
    normalization_constant
    max_logit
    probability
    expression
    label
end
function show(io::IO, ::MIME"text/plain", p::SymbolicTransformer.Prediction)
    probability = round(100*p.probability,digits=2)
    if (get(io, :compact, false) == true)
        print(io, "Prediction($probability% $(p.label)")
    else
        print(io, "Prediction($(round(100*p.probability,digits=2))% \"$(p.label)\", $(p.expression)")
    end
end

"tokenizes the utterance, and returns an operation"
function prompt(causal_lm_model::Transformers.HuggingFace.HGFGPTNeoXForCausalLM,
        encoder,
        utterance)
    model = causal_lm_model.model
    unembed = causal_lm_model.cls
    embed = model.embed
    
    tokens = encode(encoder, utterance).token

    global current_transformer = PromptedTransformer(model, encoder, embed, unembed, utterance, tokens, :(T))
    return current_transformer
end

"tokenizes the utterance, and returns a Vector of Residuals representing the embedding vectors"
function embed(transformer, utterance)    
    tokens = encode(transformer.encoder, utterance).token
    labels = decode(transformer.encoder,tokens)
    vectors = transformer.embed_layer((; token=tokens))
    expressions = map(x -> :(embed($x)), labels)
    residuals = map(x -> 
        HGFResidual(vectors.hidden_state[:,x],
            expressions[x], 
            labels[x]), 
        1:length(labels))
    return residuals
end

function embed(utterance)
    return embed(current_transformer, utterance)
end

"tokenizes the utterance, and returns a Vector of Residuals which map output residuals to logits"
function unembed(transformer, utterance::AbstractString)    
    tokens = encode(transformer.encoder, utterance).token
    labels =  decode(transformer.encoder,tokens)
    tokenids = reinterpret(Int32, tokens)
    output_vectors = transformer.unembed_layer.layer.embed.embeddings[:,tokenids]
    
    expressions = map(x -> :(unembed($x)), labels)
    residuals = map(x -> 
        HGFResidual(output_vectors[:,x],
            expressions[x], 
            labels[x]), 
        1:length(labels))
    return residuals
end

function unembed(utterance::AbstractString)
    return unembed(current_transformer, utterance)
end

function unembed(transformer, token_id::Integer)
    token_string = decode(transformer.encoder, token_id)

    return HGFResidual(transformer.unembed_layer.layer.embed.embeddings[:,token_id], :(unembed($token_string)), token_string) 
end

"applies the model to the token"
function Base.:(*)(T::PromptedTransformer, r:: HGFResidual)
    #To transform a new token at the end of a batch of tokens, we would push! the index of the 
    #new token onto tokens.onehots, which applies a corresponding change to the tokens OneHotArray
    
    #We pass in an arbitrary residual vector, so bypass the embedding layer
    input = (; token=T.tokens)
    residuals = T.embed_layer(input)
    hidden_state = hcat(residuals.hidden_state, r.vector)
    y = T.model.decoder((; hidden_state=hidden_state))
    #take the residual in the last position
    return HGFResidual(y.hidden_state[:,end], :($(T.expression) * $(r.expression)), string(T.prompt, r.label))
    
end

function LinearAlgebra.dot(r1:: HGFResidual, r2:: HGFResidual)
    return HGFResidual(r1.vector .* r2.vector, :(r1.expression ⋅ r2.expression), """< "$(r1.label)" | "$(r2.label)" >""")
end

function normalization_constant(logits)
    return sum(exp.(logits))
end

function predict(T::PromptedTransformer,r:: HGFResidual)
    "Accepts a residual which represents output from the last position in the last block of a transformer, and returns 
    predictions for the next token. The returned predictions encapsulate the logit, normalized probability, and an expression 
    which traces the tokens involved in the prediction"
    (_, logits) = T.unembed_layer((; hidden_state=r.vector))
    maxl = maximum(logits)
    shift_logits = logits .- maxl
    nc = normalization_constant(shift_logits)
    
    result = [
        begin
            probability = exp(logit-maxl) / nc
            unembed_residual = unembed(T, token_id)        
            expression = :($(unembed_residual.expression) ⋅ $(r.expression))
            label = unembed_residual.label
            Prediction(token_id, logit, nc, maxl, probability, expression, label)
        end
        for (token_id, logit) in enumerate(logits)
    ]
    #reorder by decreasing logit value
    return sort!(result; by = x -> x.logit, rev=true, dims=1)
    
end


end