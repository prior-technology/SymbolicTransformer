module HuggingfaceTransformer

    using PythonCall

    export use_pythia_70m
    export load_gptneox

    function load_gptneox(model_name, revision, cache_dir="./models/pythia-70m-deduped/step3000")
      GPTNeoXForCausalLM = pyimport("transformers" => "GPTNeoXForCausalLM");
      AutoTokenizer = pyimport("transformers" => "AutoTokenizer")
      
      model = GPTNeoXForCausalLM.from_pretrained(model_name,revision=revision,cache_dir=cache_dir)
      
      tokenizer = AutoTokenizer.from_pretrained(model_name,revision=revision,cache_dir=cache_dir)
      return (model,tokenizer)
    end

    """
    Call a model loaded from Huggingface to verify install. This is based on instructions in https://huggingface.co/EleutherAI/pythia-6.9b#quickstart 
    """
    function use_pythia_70m()
      (model,tokenizer) = load_gptneox("EleutherAI/pythia-70m-deduped","step3000")
      inputs = tokenizer("Hello, I am", return_tensors="pt")
      tokens = model.generate(input_ids = inputs.input_ids, attention_mask =inputs.attention_mask)
      tokenizer.decode(tokens[0])
    end

end

