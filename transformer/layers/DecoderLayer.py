from torch import nn

from transformer.sub_layers.PositionWiseFeedForward import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    """
    <Processing steps>
      1. Self-attention on target sequence: The input x is processed through a self-attention mechanism.
      2. Add and normalize (after self-attention): The output from self-attention is added to the original x, followed
      by dropout and normalization using norm1.
      3. Cross-attention with encoder output: The normalized output from the previous step is processed through a
      cross-attention mechanism that attends to the encoder's output enc_output.
      4. Add and normalize (after cross-attention): The output from cross-attention is added to the input of this stage,
      followed by dropout and normalization using norm2.
      5. Feed-forward network: The output from the previous step is passed through the feed-forward network.
      6. Add and normalize (after feed-forward): The feed-forward output is added to the input of this stage, followed
      by dropout and normalization using norm3.
      7. Output: The processed tensor is returned as the output of the decoder layer.
    """
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
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads)
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