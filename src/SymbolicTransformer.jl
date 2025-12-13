module SymbolicTransformer

# Include submodules
include("LayerNormalization.jl")
include("Interface.jl")
include("PythonBridge.jl")

# Re-export Interface module types
using .Interface
export Interface
export Ref, VectorRef, ScalarRef
export TokenRef, EmbeddingRef, UnembeddingRef, LayerNormRef, ResidualRef, BlockContribRef
export VectorExpr, ScalarExpr
export Embedding, Unembedding, Residual, BlockContrib, Sum, Scaled, LayerNorm, InnerProduct
export expand, evaluate, display_expr

# Re-export PythonBridge
using .PythonBridge
export PythonBridge
export connect, analyze, decompose, top_predictions

end
