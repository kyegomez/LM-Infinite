import torch
from infinite.main import LMInfinite

d_model = 512
seq_len = 100
n_global = 100
l_pretrain = 50


#sample
q = torch.randn(1, seq_len, d_model)
k = torch.randn(1, seq_len, d_model)
v = torch.randn(1, seq_len, d_model)


#llm infinite mode
model = LMInfinite(
    d_model,
    n_global,
    l_pretrain
)

#forwad pass
output = model(q, k, v)
print(output.shape)