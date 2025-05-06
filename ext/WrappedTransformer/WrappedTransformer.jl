module WrappedTransformer
using SymbolicTransformer
using Transformers
using Transformers.Layers
using Transformers.TextEncoders
using Transformers.HuggingFace
import Base.show

function SymbolicTransformer.init_context((encoder, model))
   return SymbolicTransformer.init_context(encoder, model)
end

struct GptNeoXTransformerContext <: SymbolicTransformer.TransformerLanguageModel
    encoder
    model
end

function SymbolicTransformer.init_context(encoder, causal_lm_model::Transformers.HuggingFace.HGFGPTNeoXForCausalLM)
    return GptNeoXTransformerContext(encoder, causal_lm_model)    
end

function SymbolicTransformer.init_context(encoder,causal_lm_model::Transformers.HuggingFace.HGFGPTNeoXModel)
    return GptNeoXTransformerContext(encoder, causal_lm_model.model)    
end
function SymbolicTransformer.prompt(
    model::GptNeoXTransformerContext,
    utterance::String)::SymbolicTransformer.Transformation
    
    initialisation = quote
        token_ids = ctx.tokenizer($utterance, return_tensors="pt")
        e = model.get_input_embeddings()
        embedded = e.forward(inputs.input_ids)
    end
    
    transform = quote
        output = model.gpt_neox(inputs_embeds=embedded)
    end

    return SymbolicTransformer.Transformation(
        model,
        initialisation,
        transform,
        Expr(:empty)
    )
end

end