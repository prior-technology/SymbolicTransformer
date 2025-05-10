module WrappedTransformer
using SymbolicTransformer
using Transformers
using Transformers.Layers
using Transformers.TextEncoders
using Transformers.HuggingFace
import Base.show

function SymbolicTransformer.init_context((encoder, model)::Tuple)
   return SymbolicTransformer.init_context(encoder, model)
end

struct GptNeoXTransformerContext <: SymbolicTransformer.TransformerLanguageModel
    encoder
    model
    embed_layer
    unembed_layer
end
function SymbolicTransformer.init_context(model_name::AbstractString)
    config = Transformers.HuggingFace.load_config(model_name)
    tokenizer = load_tokenizer(model_name, config=config)
    model = Transformers.HuggingFace.load_model(config.model_type, model_name, "forcausallm"; config=config)
    return SymbolicTransformer.init_context(tokenizer, model)
end

function SymbolicTransformer.init_context(encoder, causal_lm_model::Transformers.HuggingFace.HGFGPTNeoXForCausalLM)
    return SymbolicTransformer.init_context(encoder, causal_lm_model.model, causal_lm_model.cls)
end

function SymbolicTransformer.init_context(encoder,model::Transformers.HuggingFace.HGFGPTNeoXModel, unembed)
    return GptNeoXTransformerContext(encoder, model, model.embed_layer, unembed)    
end
function SymbolicTransformer.prompt(
    context::GptNeoXTransformerContext,
    utterance::String)::SymbolicTransformer.Transformation
    
    tokens = Transformers.TextEncoders.encode(context.encoder, utterance).token
    labels = decode(context.encoder,tokens)
    last_token_label = labels[end]
    context.encoder.encode

    initialisation = quote
        tokens = $labels
        T(vector) = model.forward(vector)
    end
    
    transform = quote

        output = T(vector)
    end

    return SymbolicTransformer.Transformation(
        model,
        initialisation,
        transform,
        Expr(:empty)
    )
end


end