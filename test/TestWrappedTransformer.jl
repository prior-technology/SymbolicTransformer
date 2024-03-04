using Transformers.HuggingFace
using Transformers.TextEncoders
using SymbolicTransformer

encoder, model = hgf"EleutherAI/pythia-14m"

@testset "prompt" begin
    T = SymbolicTransformer.WrappedTransformer.prompt(model, encoder, "Hello, world!")    
    
    @test p.prompt == "Hello, world!"
end
