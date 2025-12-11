"""
Interface types for SymbolicTransformer ⟷ TransformerAlgebra communication.

This module defines Julia-side reference types and expression types that
mirror the Python interface. These enable symbolic manipulation in Julia
while delegating actual model operations to Python.
"""

module Interface

using LinearAlgebra
using JSON3

export Ref, VectorRef, ScalarRef
export TokenRef, EmbeddingRef, UnembeddingRef, LayerNormRef, ResidualRef, BlockContribRef
export Expr, VectorExpr, ScalarExpr
export Embedding, Unembedding, Residual, BlockContrib, Sum, Scaled, LayerNorm, InnerProduct
export expand, evaluate, display_expr

# =============================================================================
# Reference Types (mirror Python interface types)
# =============================================================================

abstract type Ref end

"""Reference to a token in the vocabulary."""
struct TokenRef <: Ref
    token_id::Int
    text::String
end

"""Reference to an embedding vector (row of W_E)."""
struct EmbeddingRef <: Ref
    token::TokenRef
    weights_path::Vector{Union{String, Int}}
end

"""Reference to an unembedding vector (row of W_U)."""
struct UnembeddingRef <: Ref
    token::TokenRef
    weights_path::Vector{Union{String, Int}}
end

"""Reference to layer normalization parameters."""
struct LayerNormRef <: Ref
    weight_path::Vector{Union{String, Int}}
    bias_path::Vector{Union{String, Int}}
    epsilon::Float64
end

"""Reference to a residual vector in a cached context."""
struct ResidualRef <: Ref
    context_id::String
    layer::Int
    position::Int
end

"""Reference to a transformer block's contribution."""
struct BlockContribRef <: Ref
    context_id::String
    block::Int
    position::Int
    component::String  # "attention", "mlp", or "total"
    head::Union{Int, Nothing}
end

# Constructors with defaults
BlockContribRef(ctx, block, pos, comp) = BlockContribRef(ctx, block, pos, comp, nothing)

# =============================================================================
# JSON Serialization for References
# =============================================================================

function to_dict(ref::TokenRef)
    Dict("type" => "TokenRef", "token_id" => ref.token_id, "text" => ref.text)
end

function to_dict(ref::EmbeddingRef)
    Dict(
        "type" => "EmbeddingRef",
        "token" => to_dict(ref.token),
        "weights_path" => Dict("type" => "ModelPath", "segments" => ref.weights_path)
    )
end

function to_dict(ref::UnembeddingRef)
    Dict(
        "type" => "UnembeddingRef",
        "token" => to_dict(ref.token),
        "weights_path" => Dict("type" => "ModelPath", "segments" => ref.weights_path)
    )
end

function to_dict(ref::LayerNormRef)
    Dict(
        "type" => "LayerNormRef",
        "weight_path" => Dict("type" => "ModelPath", "segments" => ref.weight_path),
        "bias_path" => Dict("type" => "ModelPath", "segments" => ref.bias_path),
        "epsilon" => ref.epsilon
    )
end

function to_dict(ref::ResidualRef)
    Dict(
        "type" => "ResidualRef",
        "context_id" => ref.context_id,
        "layer" => ref.layer,
        "position" => ref.position
    )
end

function to_dict(ref::BlockContribRef)
    Dict(
        "type" => "BlockContribRef",
        "context_id" => ref.context_id,
        "block" => ref.block,
        "position" => ref.position,
        "component" => ref.component,
        "head" => ref.head
    )
end

to_json(ref::Ref) = JSON3.write(to_dict(ref))

# =============================================================================
# Expression Types (for symbolic manipulation)
# =============================================================================

abstract type Expr end
abstract type VectorExpr <: Expr end
abstract type ScalarExpr <: Expr end

"""Embedding vector expression."""
struct Embedding <: VectorExpr
    ref::EmbeddingRef
    label::String
end
Embedding(ref::EmbeddingRef) = Embedding(ref, ref.token.text)

"""Unembedding vector expression."""
struct Unembedding <: VectorExpr
    ref::UnembeddingRef
    label::String
end
Unembedding(ref::UnembeddingRef) = Unembedding(ref, ref.token.text)

"""Residual vector expression."""
struct Residual <: VectorExpr
    ref::ResidualRef
    label::String
end
function Residual(ref::ResidualRef)
    Residual(ref, "x$(subscript(ref.position))$(superscript(ref.layer))")
end

"""Block contribution expression."""
struct BlockContrib <: VectorExpr
    ref::BlockContribRef
    label::String
end
function BlockContrib(ref::BlockContribRef)
    BlockContrib(ref, "Δx$(subscript(ref.position))$(superscript(ref.block))")
end

"""Sum of vector expressions."""
struct Sum <: VectorExpr
    terms::Vector{VectorExpr}
end

"""Scaled vector expression."""
struct Scaled <: VectorExpr
    scalar::Union{Float64, ScalarExpr}
    vector::VectorExpr
end

"""Layer-normalized vector expression."""
struct LayerNorm <: VectorExpr
    ln_ref::LayerNormRef
    input::VectorExpr
end

"""Bias vector from layer norm."""
struct LNBias <: VectorExpr
    ln_ref::LayerNormRef
end

"""Centered vector (x - mean(x))."""
struct Centered <: VectorExpr
    input::VectorExpr
end

"""Gamma-scaled vector (γ ⊙ x)."""
struct GammaScaled <: VectorExpr
    ln_ref::LayerNormRef
    input::VectorExpr
end

"""Inner product of two vector expressions."""
struct InnerProduct <: ScalarExpr
    left::VectorExpr
    right::VectorExpr
end

"""Sum of scalar expressions."""
struct ScalarSum <: ScalarExpr
    terms::Vector{ScalarExpr}
end

"""Scaled scalar expression."""
struct ScalarScaled <: ScalarExpr
    scale::Float64
    expr::ScalarExpr
end

# =============================================================================
# Operators
# =============================================================================

Base.:(+)(a::VectorExpr, b::VectorExpr) = Sum([a, b])
Base.:(+)(s::Sum, b::VectorExpr) = Sum([s.terms..., b])
Base.:(+)(a::VectorExpr, s::Sum) = Sum([a, s.terms...])
Base.:(+)(s1::Sum, s2::Sum) = Sum([s1.terms..., s2.terms...])

Base.:(*)(s::Number, v::VectorExpr) = Scaled(Float64(s), v)
Base.:(*)(v::VectorExpr, s::Number) = Scaled(Float64(s), v)

LinearAlgebra.dot(a::VectorExpr, b::VectorExpr) = InnerProduct(a, b)

Base.:(+)(a::ScalarExpr, b::ScalarExpr) = ScalarSum([a, b])
Base.:(*)(s::Number, e::ScalarExpr) = ScalarScaled(Float64(s), e)

# =============================================================================
# Display Helpers
# =============================================================================

function subscript(n::Int)
    if n < 0
        return "₋" * subscript(-n)
    end
    digits = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
    n == 0 && return "₀"
    result = ""
    while n > 0
        result = digits[(n % 10) + 1] * result
        n = n ÷ 10
    end
    return result
end

function superscript(n::Int)
    if n < 0
        return "⁻" * superscript(-n)
    end
    digits = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹']
    n == 0 && return "⁰"
    result = ""
    while n > 0
        result = digits[(n % 10) + 1] * result
        n = n ÷ 10
    end
    return result
end

# Display methods
display_expr(e::Embedding) = "$(e.label)̲"  # underline for embedding
display_expr(e::Unembedding) = "$(e.label)̄"  # overline for unembedding
display_expr(e::Residual) = e.label
display_expr(e::BlockContrib) = e.label
display_expr(e::LNBias) = "β"
display_expr(e::Centered) = "c($(display_expr(e.input)))"
display_expr(e::GammaScaled) = "γ⊙$(display_expr(e.input))"

function display_expr(e::Sum)
    terms = join([display_expr(t) for t in e.terms], " + ")
    return "($terms)"
end

function display_expr(e::Scaled)
    if e.scalar isa Float64
        return "$(round(e.scalar, digits=3))·$(display_expr(e.vector))"
    else
        return "$(display_expr(e.scalar))·$(display_expr(e.vector))"
    end
end

function display_expr(e::LayerNorm)
    return "LN($(display_expr(e.input)))"
end

function display_expr(e::InnerProduct)
    return "⟨$(display_expr(e.left)), $(display_expr(e.right))⟩"
end

function display_expr(e::ScalarSum)
    terms = join([display_expr(t) for t in e.terms], " + ")
    return "($terms)"
end

function display_expr(e::ScalarScaled)
    return "$(round(e.scale, digits=3))·$(display_expr(e.expr))"
end

Base.show(io::IO, e::Expr) = print(io, display_expr(e))

# =============================================================================
# Expansion Rules
# =============================================================================

"""
Expand a residual expression into embedding + block contributions.

x^L_j = embed_j + Δx^1_j + Δx^2_j + ... + Δx^L_j
"""
function expand(r::Residual, n_layers::Int)
    ctx = r.ref.context_id
    pos = r.ref.position

    # Embedding is layer 0
    embed_ref = ResidualRef(ctx, 0, pos)
    embed = Residual(embed_ref, "embed$(subscript(pos))")

    # Block contributions
    contribs = [
        BlockContrib(BlockContribRef(ctx, i, pos, "total"))
        for i in 1:n_layers
    ]

    return Sum([embed, contribs...])
end

"""
Expand an inner product through layer norm.

⟨y, LN(a + b)⟩ = scale · (⟨γ⊙y, c(a)⟩ + ⟨γ⊙y, c(b)⟩) + ⟨y, β⟩

where c(x) = x - mean(x) and scale = 1/||a + b - mean||
"""
function expand(ip::InnerProduct)
    if ip.right isa LayerNorm && ip.right.input isa Sum
        ln = ip.right
        terms = ln.input.terms

        # Each term becomes centered and gamma-scaled
        expanded = [
            InnerProduct(
                GammaScaled(ln.ln_ref, ip.left),
                Centered(term)
            )
            for term in terms
        ]

        # Add bias term
        push!(expanded, InnerProduct(ip.left, LNBias(ln.ln_ref)))

        # Return as sum (scale factor needs to be computed at evaluation time)
        return ScalarSum(expanded)
    end
    return ip
end

"""
Expand by applying all applicable expansion rules recursively.
"""
function expand_all(e::Expr, n_layers::Int)
    if e isa Residual
        return expand_all(expand(e, n_layers), n_layers)
    elseif e isa InnerProduct
        expanded = expand(e)
        if expanded !== e
            return expand_all(expanded, n_layers)
        end
    end
    return e
end

# =============================================================================
# Python Service Bridge
# =============================================================================

# This would typically use PythonCall.jl to communicate with Python.
# For now, define the interface that implementations should provide.

abstract type PythonBridge end

"""
Evaluate an expression by calling Python to resolve references.

Implementations should:
1. Convert the expression to reference(s)
2. Call Python's TransformerService to resolve/compute
3. Return the numeric result
"""
function evaluate(bridge::PythonBridge, expr::Expr)
    error("evaluate not implemented for $(typeof(bridge))")
end

"""
Create a context in Python and return the context_id.
"""
function create_context(bridge::PythonBridge, prompt::String)
    error("create_context not implemented for $(typeof(bridge))")
end

"""
Get context info (tokens, n_layers, etc.) from Python.
"""
function get_context_info(bridge::PythonBridge, context_id::String)
    error("get_context_info not implemented for $(typeof(bridge))")
end

end # module
