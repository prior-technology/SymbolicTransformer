using SymbolicTransformer

module SymbolCreator

function TransformerLensCacheParser(cache)
    
    function GetResidual(position, layer)
        return cache["blocks.{layer}.hook_resid_pre"]
    end

end

end