"""
PythonCall-based bridge to TransformerAlgebra.

This module provides the actual implementation that calls Python's
TransformerService to resolve references and perform computations.
"""

module PythonBridge

using PythonCall
using ..Interface

export PythonCallBridge, connect, create_context, get_context_info
export evaluate, resolve, inner_product, decompose_logit

# =============================================================================
# Bridge Type
# =============================================================================

"""
Bridge to Python's TransformerService using PythonCall.
"""
mutable struct PythonCallBridge <: Interface.PythonBridge
    service::Py  # Python TransformerService instance
    model::Py    # Python model (for reference)
    tokenizer::Py
end

# =============================================================================
# Connection
# =============================================================================

"""
Connect to Python and create a TransformerService.

Example:
    bridge = connect("EleutherAI/pythia-160m-deduped")
"""
function connect(model_name::String="EleutherAI/pythia-160m-deduped")
    # Import Python modules
    ta = pyimport("transformer_algebra")

    # Load model
    model, tokenizer = ta.load_pythia_model(model_name)

    # Create service
    service = ta.TransformerService(model, tokenizer)

    return PythonCallBridge(service, model, tokenizer)
end

# =============================================================================
# Context Management
# =============================================================================

"""
Create a context (run model on prompt, cache states).
Returns context_id.
"""
function create_context(bridge::PythonCallBridge, prompt::String)
    context_id = bridge.service.create_context(prompt)
    return pyconvert(String, context_id)
end

"""
Get context info (tokens, dimensions, etc).
"""
function get_context_info(bridge::PythonCallBridge, context_id::String)
    info = bridge.service.get_context_info(context_id)
    return pyconvert(Dict, info)
end

# =============================================================================
# Reference Resolution
# =============================================================================

"""
Resolve a reference to its tensor value (as Julia Vector{Float64}).
"""
function resolve(bridge::PythonCallBridge, ref::Interface.Ref)
    ref_dict = Interface.to_dict(ref)
    result = bridge.service.resolve_to_list(ref_dict)
    return pyconvert(Vector{Float64}, result)
end

# =============================================================================
# Computations
# =============================================================================

"""
Compute inner product of two references.
"""
function inner_product(bridge::PythonCallBridge, ref1::Interface.Ref, ref2::Interface.Ref)
    dict1 = Interface.to_dict(ref1)
    dict2 = Interface.to_dict(ref2)
    result = bridge.service.inner_product(dict1, dict2)
    return pyconvert(Float64, result)
end

"""
Get top k predictions from a residual.
"""
function top_predictions(bridge::PythonCallBridge, residual_ref::Interface.ResidualRef, k::Int=10)
    ref_dict = Interface.to_dict(residual_ref)
    result = bridge.service.top_predictions(ref_dict, k)
    return pyconvert(Vector{Dict}, result)
end

"""
Get logit for a specific token.
"""
function logit_for_token(bridge::PythonCallBridge, residual_ref::Interface.ResidualRef, token_text::String)
    ref_dict = Interface.to_dict(residual_ref)
    result = bridge.service.logit_for_token(ref_dict, token_text)
    return pyconvert(Float64, result)
end

"""
Decompose a logit into per-block contributions.
"""
function decompose_logit(bridge::PythonCallBridge, context_id::String, token_text::String, position::Int=-1)
    result = bridge.service.decompose_logit(context_id, token_text, position)
    return pyconvert(Vector{Dict}, result)
end

# =============================================================================
# Expression Evaluation
# =============================================================================

"""
Evaluate an expression by resolving references through Python.
"""
function evaluate(bridge::PythonCallBridge, expr::Interface.Embedding)
    return resolve(bridge, expr.ref)
end

function evaluate(bridge::PythonCallBridge, expr::Interface.Unembedding)
    return resolve(bridge, expr.ref)
end

function evaluate(bridge::PythonCallBridge, expr::Interface.Residual)
    return resolve(bridge, expr.ref)
end

function evaluate(bridge::PythonCallBridge, expr::Interface.BlockContrib)
    return resolve(bridge, expr.ref)
end

function evaluate(bridge::PythonCallBridge, expr::Interface.InnerProduct)
    # If both sides are simple refs, compute directly
    if expr.left isa Interface.Unembedding && expr.right isa Interface.Residual
        return inner_product(bridge, expr.left.ref, expr.right.ref)
    end

    # Otherwise evaluate both sides and compute locally
    left_vec = evaluate(bridge, expr.left)
    right_vec = evaluate(bridge, expr.right)
    return sum(left_vec .* right_vec)
end

function evaluate(bridge::PythonCallBridge, expr::Interface.Sum)
    result = zeros(Float64, length(evaluate(bridge, expr.terms[1])))
    for term in expr.terms
        result .+= evaluate(bridge, term)
    end
    return result
end

function evaluate(bridge::PythonCallBridge, expr::Interface.Scaled)
    vec = evaluate(bridge, expr.vector)
    if expr.scalar isa Float64
        return expr.scalar .* vec
    else
        scale = evaluate(bridge, expr.scalar)
        return scale .* vec
    end
end

# =============================================================================
# High-Level Analysis Functions
# =============================================================================

"""
Analyze a prompt and return an object for interactive exploration.

Example:
    bridge = connect()
    analysis = analyze(bridge, "The capital of Ireland")
    top_predictions(analysis)
    decompose(analysis, " Dublin")
"""
struct PromptAnalysis
    bridge::PythonCallBridge
    context_id::String
    prompt::String
    tokens::Vector{Dict}
    n_layers::Int
    d_model::Int
end

function analyze(bridge::PythonCallBridge, prompt::String)
    context_id = create_context(bridge, prompt)
    info = get_context_info(bridge, context_id)

    return PromptAnalysis(
        bridge,
        context_id,
        prompt,
        info["tokens"],
        info["n_layers"],
        info["d_model"]
    )
end

function top_predictions(analysis::PromptAnalysis, position::Int=-1, k::Int=10)
    ref = Interface.ResidualRef(analysis.context_id, analysis.n_layers, position)
    return top_predictions(analysis.bridge, ref, k)
end

function decompose(analysis::PromptAnalysis, token_text::String, position::Int=-1)
    return decompose_logit(analysis.bridge, analysis.context_id, token_text, position)
end

"""
Get the residual at a specific layer and position.
"""
function residual(analysis::PromptAnalysis, layer::Int, position::Int=-1)
    return Interface.Residual(Interface.ResidualRef(analysis.context_id, layer, position))
end

"""
Get the final residual (after last block).
"""
function final_residual(analysis::PromptAnalysis, position::Int=-1)
    return residual(analysis, analysis.n_layers, position)
end

"""
Get unembedding vector for a token.
"""
function unembed(analysis::PromptAnalysis, token_text::String)
    token_ref = Interface.TokenRef(0, token_text)  # token_id filled by Python
    # Actually need to call Python to get the real token_id
    token_refs = pyconvert(Vector{Dict}, analysis.bridge.service.tokenize(token_text))
    if length(token_refs) != 1
        error("'$token_text' tokenizes to $(length(token_refs)) tokens, expected 1")
    end
    tr = token_refs[1]
    token = Interface.TokenRef(tr["token_id"], tr["text"])

    # Get unembedding path from profile
    path = pyconvert(Vector, analysis.bridge.service.profile.unembedding_weights.segments)
    return Interface.Unembedding(Interface.UnembeddingRef(token, path))
end

"""
Build a logit expression: ⟨token̄, LN(x)⟩
"""
function logit_expr(analysis::PromptAnalysis, token_text::String, position::Int=-1)
    unemb = unembed(analysis, token_text)
    resid = final_residual(analysis, position)

    # Get final LN ref
    ln_ref = Interface.LayerNormRef(
        pyconvert(Vector, analysis.bridge.service.profile.final_ln_weight.segments),
        pyconvert(Vector, analysis.bridge.service.profile.final_ln_bias.segments),
        pyconvert(Float64, analysis.bridge.service.profile.final_ln_epsilon)
    )

    return Interface.InnerProduct(unemb, Interface.LayerNorm(ln_ref, resid))
end

end # module
