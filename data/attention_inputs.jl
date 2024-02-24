# based on 
# import torch
# import numpy as np
# import pandas as pd

# from transformer_lens import HookedTransformer
# model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped")
# logits, activations = model.run_with_cache("Two words")
# q = activations["blocks.0.attn.hook_q"][0,:,0,:]
# k = activations["blocks.0.attn.hook_k"][0,:,0,:]
# a = activations["blocks.0.attn.hook_attn_scores"][0,0]

# pd.DataFrame(q.numpy()).to_csv("q.csv", index=False)
# pd.DataFrame(k.numpy()).to_csv("k.csv", index=False)
# pd.DataFrame(a.numpy()).to_csv("a.csv", index=False)
#
#remove first line from each csv

using DelimitedFiles
folder = dirname(@__FILE__)
q = readdlm(joinpath(folder,"q.csv"), ',', Float16, '\n')
k = readdlm(joinpath(folder,"k.csv"), ',', Float16, '\n')

expected_attention =  readdlm(joinpath(folder,"a.csv"), ',', Float16, '\n')

