[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# LM-INFINITE: SIMPLE ON-THE-FLY LENGTH GENERALIZATION FOR LARGE LANGUAGE MODELS

LM-Infinite is a solution proposed by Chi Han, Qifan Wang, Wenhan Xiong, Yu Chen, Heng Ji, and Sinong Wang to address the length generalization failure of Large Language Models (LLMs) on long sequences. LLMs, such as Transformer-based models, have shown impressive performance in various domains but struggle when it comes to longer reasoning processes or understanding larger contexts. Current pre-training schemes truncate training sequences to a fixed length, and even with relative positional encoding, LLMs struggle to generate coherent texts or perform downstream tasks after longer contexts.

The authors investigate the main out-of-distribution factors contributing to this problem and propose LM-Infinite as an efficient solution. LM-Infinite only requires a Λ-shaped attention mask and a distance limit, without any parameter updates or learning. It can be applied to different LLMs using relative-position encoding methods. LM-Infinite demonstrates consistent fluency and generation quality for sequences as long as 32k tokens on datasets like ArXiv and OpenWebText2, with a decoding speedup of 2.72x. Furthermore, it continues to perform well on inputs much longer than training lengths in downstream tasks like passkey retrieval, where vanilla models fail immediately.

[Paper Link](https://arxiv.org/pdf/2308.16137.pdf)

---

# Appreciation
* Lucidrains
* Agorians



# Install
`pip install lm-infinite`

# Usage
```python
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
```
# Architecture

# Todo


# License
MIT

# Citations
```latex
@misc{2308.16137,
Author = {Chi Han and Qifan Wang and Wenhan Xiong and Yu Chen and Heng Ji and Sinong Wang},
Title = {LM-Infinite: Simple On-the-Fly Length Generalization for Large Language Models},
Year = {2023},
Eprint = {arXiv:2308.16137},
}
```