import math

import torch
from torch import nn

# PositionalEncoding is a subclass of PyTorch's nn.Module, allowing it to be used as a standard PyTorch layer
class PositionalEncoding(nn.Module):
    # pe is registered as a buffer - it will be part of the module's state but will not be considered a trainable
    # parameter
    def __init__(self, d_model, max_seq_length):
        """
        Args:
            d_model: model input dimension
            max_seq_length: max sequence length of the sequence for which positional encodings are pre-computed
        Attributes:
            pe: A tensor filed with zeros, which will be populated with positional encodings
            position: A tensor containing the position indices for each position in the sequence
            div_term: A term used to scale the position indices in a specific way
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # The sine function is applied to the even indices & the cosine function to the odd indices of pe
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # The forward method simply adds the positional encodings to the input x.
        return x + self.pe[:, :x.size(1)]