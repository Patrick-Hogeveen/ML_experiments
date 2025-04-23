import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MACTitan(nn.Module):
    def __init__(
            self,
            num_tokens,
            dims,
            depth,
            segment_len,
            hdim=64,
            nheads=8,
            ff_mult=4,
        
    ):
        token_emb = nn.Embedding(num_tokens, dims)
        self.token_emb = token_emb