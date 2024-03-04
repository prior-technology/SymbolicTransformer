
using Transformers.HuggingFace
using SymbolicTransformer
encoder, model = hgf"EleutherAI/pythia-14m"

@testset "prompt" begin
    p = SymbolicTransformer.WrappedTransformer.prompt(model, encoder, "Hello, world!")
    @test p.prompt == "Hello, world!"
end
