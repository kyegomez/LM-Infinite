import torch
from infinite.main import LMInfinite


lm_infinite = LMInfinite(
    n_global=10,
    n_local=2048,
    length_pretrain=2048
)

#sample data
q = torch.rand(10, 32, 512) #shape = [sequence length, batch_size, feature_size]
k = torch.rand(10, 32, 512) #shape = [sequence_length, batch_size, feature_size]

#apply lambda attention mask and distance limit
logit = lm_infinite(q, k)

print(logit)