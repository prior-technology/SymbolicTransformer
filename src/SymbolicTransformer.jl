module SymbolicTransformer

abstract type TransformerLanguageModel end
struct Transformation
    context::TransformerLanguageModel
    setup::Expr
    transform::Expr
    interpretation::Expr
end

function prompt(model::TransformerLanguageModel, input::String)::Transformation
    "Abstract function to prompt the model with a string input."
    # This function should be implemented in the model-specific extension module
    # and should return a Transform object that maps an embedded vector to an output.
    return Transformation(
        empty,
        Expr(:empty),
        Expr(:empty),
        Expr(:empty)
    )
end

function init_context()
    "Initialize the context with implementation specific model and tokenizer."
    
    return error("init_context() not implemented for this model.")
end

export init_context

end