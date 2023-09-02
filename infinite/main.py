import torch
from torch import nn


class LMInfinite(nn.Module):
    def __init__(
        self,
        *,
        n_global,
        n_local,
        length_pretrain
    ):
        super(LMInfinite, self).__init__()
        self.n_global = n_global
        self.n_local = n_local
        self.length_pretrain = length_pretrain
    
    def lambda_attention_mask(self, input_tokens):
        #tensor to hold attention mask
        mask = torch.zeros(
            input_tokens.shape[0],
            input_tokens.shape[0]
        )

        #global branch of the mask
        mask[:self.n_local, :] = 1

        #create the local branch of the mask
        for i in range(self.n_local, input_tokens.shape[0]):
            mask[i, i-self.n_local:i] = 1

        return mask
    
    def distance_limit(self, q, k):
        #cal distance between pair of tokens
        d = torch.abs(torch.arange(q.shape[0])).unsqueeze(0) - torch.arange(k.shape[0]).unsqueeze(1)

        #distance limit
        d = torch.clamp(d, max=self.length_pretrain)

        #cal attention logit
        logit = torch.bmm(
            q, 
            k.transpose(1, 2) // \
            (q.shape[-1] ** 0.5)
        )

        #modify attention logit
        logit = logit / torch.max(
            d,
            dim=-1,
            keepdim=True
        ).values.unsqueeze(-1) # add extra dimension

        return logit
    

    def forward(self, q, k):
        #apply the lambda attention mask
        mask = self.lambda_attention_mask(q)

        #distance limit
        logit = self.distance_limit(q, k)

        #mask to logit
        logit = logit * mask#.unsqueeze(-1)

        return logit
    

