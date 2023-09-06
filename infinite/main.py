import torch
from torch import nn
import math

import torch.nn.functional as F

class LMInfinite(nn.Module):
    def __init__(
        self,
        d_model,
        n_global=100,
        l_pretrain=2048
        ):
        super(LMInfinite, self).__init__()
        self.d_model = d_model
        self.n_global = n_global
        self.l_pretrain=l_pretrain

    def lambda_mask(self, seq_len):
        #create mask of shape (seq_len, seq_len) with ones on the allowed positions and negative infinite on the disallowed positions
        mask = torch.full((seq_len, seq_len), float('-inf'))
        for i in range(seq_len):
            #global branch
            mask[i, :min(self.n_global, i+1)] = 0
            
            #local branch
            mask[i, max(0, i-self.l_pretrain+1):i+1] = 0
        return mask
    
    def distance_limit(self, distance):
        #bound the effective distance within l_pretrain
        return torch.clamp(distance, max=self.l_pretrain)
    
    def forward(self, q, k, v):
        seq_len = q.size(1)

        #compute attention logits
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)

        #compute the distances between each pair of tokens
        distances = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)

        #distance limit
        distances = self.distance_limit(distances)

        #add distance limit to the logits
        logits += distances

        #apply lambda mask
        mask = self.lambda_mask(seq_len)
        logits = logits + mask.to(logits.device)

        #attention weights
        weights = F.softmax(logits, dim=-1)

        #output
        output = torch.matmul(weights, v)
        return output