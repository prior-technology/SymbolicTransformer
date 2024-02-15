using PythonCall

module HuggingfaceTransformer

    export use_pythia_70m

    """
    Call a model loaded from Huggingface to verify install. This is based on instructions in https://huggingface.co/EleutherAI/pythia-6.9b#quickstart 
    """
    function use_pythia_70m()
        GPTNeoXForCausalLM = pyimport("transformers" => "GPTNeoXForCausalLM");
        AutoTokenizer = pyimport("transformers" => "AutoTokenizer")
        
        model = GPTNeoXForCausalLM.from_pretrained(
          "EleutherAI/pythia-70m-deduped",
          revision="step3000",
          cache_dir="./models/pythia-70m-deduped/step3000",
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
          "EleutherAI/pythia-70m-deduped",
          revision="step3000",
          cache_dir="./models/pythia-70m-deduped/step3000",
        )
        
        inputs = tokenizer("Hello, I am", return_tensors="pt")
        tokens = model.generate(input_ids = inputs.input_ids, attention_mask =inputs.attention_mask)
        tokenizer.decode(tokens[0])
    end
end

