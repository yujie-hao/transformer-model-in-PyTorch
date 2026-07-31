from torch import nn

from transformer.sub_layers.MultiHeadAttention import MultiHeadAttention
from transformer.sub_layers.PositionWiseFeedForward import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        """
        Args:
            d_model: input dimension
            num_heads: number of attention heads in the multi-head attention
            d_ff: dimensionality of the inner layer in the feed-forward network
            dropout: dropout rate for regularization
        Attributes:
            self.self_attn: Multi-head self-attention mechanism for the target sequence
            self.cross_attn: Multi-head cross-attention mechanism that attends to the encoder's output
            self.feed_forward: Position-wise feed-forward neural network
            self.norm1, self.norm2, self.norm3: Layer normalization components
            self.dropout: Dropout layer for regularization
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        """
        Args:
            x: Input sequence to the decoder layer
            enc_output: Encoder output (used in the cross-attention step)
            src_mask: Source mask to ignore certain parts of the encoder's output
            tgt_mask: Target mask to ignore certain parts of the decoder's input
        """
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x