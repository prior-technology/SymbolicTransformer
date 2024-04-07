module WrappedTransformer
using Transformers
using Transformers.Layers
using Transformers.TextEncoders
using Transformers.HuggingFace
using SymbolicTransformer
using LinearAlgebra
import Base.show

export PromptedTransformer, HGFResidual, prompt, embed, unembed, predict, dot, prompt_residuals, extract_blocks, expand

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

struct PromptedTransformerBlock <: SymbolicTransformer.Operation
    "One block of a Transformers.jl Huggingface transformer"
    block
    prompt_residuals
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

struct HGFTransformerBlock <: SymbolicTransformer.Operation
    "One block of a Transformers.jl Huggingface transformer"
    
    expression
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

function bra(s::AbstractString)
    return "⟨ $s |"
end
function ket(s::AbstractString)
    return "| $s ⟩"
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
        HGFResidual(adjoint(output_vectors[:,x]),
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


function Base.:(+)(r1:: HGFResidual, r2:: HGFResidual)
    return HGFResidual(r1.vector + r2.vector, :($(r1.expression) + $(r2.expression)), """$(r1.label) + $(r2.label)""")
end

function prompt_residuals(T::PromptedTransformer)        
    #We pass in an arbitrary residual vector, so bypass the embedding layer
    input = (; token=T.tokens)
    return T.embed_layer(input)
end

function apply(T::PromptedTransformer, hidden_state)
    T.model.decoder((; hidden_state=hidden_state))
end
function apply(B::PromptedTransformerBlock, hidden_state)
    B((; hidden_state=hidden_state))
end

function append_hidden_state(hidden_state, r::HGFResidual)
    return hcat(hidden_state, r.vector)
end
function append_hidden_state(hidden_state, target_residuals:: AbstractVector{HGFResidual})
    
    new_residual_matrix = hcat([r.vector for r in target_residuals]...)
    hcat(hidden_state, new_residual_matrix)
end
"applies the model to the token"
function Base.:(*)(T::SymbolicTransformer.Operation, r:: HGFResidual)
    #To transform a new token at the end of a batch of tokens, we would push! the index of the 
    #new token onto tokens.onehots, which applies a corresponding change to the tokens OneHotArray

    residuals = prompt_residuals(T)
    hidden_state = append_hidden_state(residuals.hidden_state, r)
    y = apply(T,hidden_state)
    #take the residual in the last position
    return HGFResidual(y.hidden_state[:,end], :($(T.expression) * $(r.expression)), string(T.prompt, r.label))
    
end
function Base.:(*)(T::SymbolicTransformer.Operation, target_residuals :: AbstractVector{HGFResidual})

    residuals = prompt_residuals(T)
    hidden_state = append_hidden_state(residuals.hidden_state, target_residuals)
    y = apply(T,hidden_state)
    
    #return output residuals in positions corresponding with the target residuals    
    result_vectors = y.hidden_state[:,end-length(target_residuals)+1:end]
    return [HGFResidual(result_vectors[:,i], :($(T.expression) * $(target_residuals[i].expression)), string(T.prompt, target_residuals[i].label)) for i in eachindex(target_residuals)]
end

function LinearAlgebra.dot(r1:: HGFResidual, r2:: HGFResidual)
    return HGFResidual(LinearAlgebra.dot(r1.vector,r2.vector), :($(r1.expression) ⋅ $(r2.expression)), """< "$(r1.label)" | "$(r2.label)" >""")
end

function LinearAlgebra.transpose(r:: HGFResidual)
    return HGFResidual(transpose(r.vector), :(transpose($(r.expression))), """ transpose($(r.label)) """)
end

function LinearAlgebra.adjoint(r:: HGFResidual)
    return HGFResidual(adjoint(r.vector), :(($(r.expression))'), """ ($(r.label))' """)
end

function LinearAlgebra.dot(v1:: Vector{HGFResidual}, v2:: Vector{HGFResidual})
    return dot.(v1, v2)
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
function wrap(ln::Transformers.Layers.LayerNorm)
    return :(LN)

end

function promptBlock(block::Transformers.Layers.AbstractTransformerBlock, residuals::AbstractVector{HGFResidual})

    #return PromptedTransformerBlock(block, residuals,:($block * $residuals))
end
function wrap(transformer_blocks::Transformers.Layers.Transformer, input_residuals::AbstractVector{HGFResidual})
    #the operations within transformer operator are composed
    #so return an expression with each operation seperated by the composition operator ∘
    return []
    
end

function prefix_block(block::Transformers.Layers.AbstractTransformerBlock, prefix_residuals)
    "Return a PromptedTransformerBlock which includes prefix_residuals with the result of applying those residuals to the block"
    promptedBlock = PromptedTransformerBlock(block, prefix_residuals, :($block * $prefix_residuals))
    residuals = block(prefix_residuals) 
    return (residuals, promptedBlock)
end

function apply_blocks(blocks, prefix_residuals)
    "Takes an iterable of Transformer blocks and an initial residual. Returns PromptedTransformerBlocks
    where each includes residuals from applying the last prefix to the last transformer"
    result = []
    for block in blocks
        (prefix_residuals, promptedBlock) = prefix_block(block, prefix_residuals)
        push!(result, promptedBlock)
    end
    return result
end
function extract_blocks(model::Transformers.HuggingFace.HGFGPTNeoXModel, prefix_residuals)
    ln = model.decoder.layers[2]
    transformer = model.decoder.layers[1]
    return (ln = ln, blocks = apply_blocks(transformer.blocks, prefix_residuals))
end

function extract_blocks(model::Transformers.HuggingFace.HGFGPTNeoXForCausalLM, prefix_residuals)    
    return extract_blocks(model.model, prefix_residuals)
end

function extract_blocks(T::PromptedTransformer)    
    residuals = prompt_residuals(T)
    return extract_blocks(T.model, residuals)
end

function expand(T::PromptedTransformer, r:: HGFResidual)
    "Replace T with the blocks of the transformer"
    blocks = T.model
end

end