from torch import nn

from transformer.sub_layers.position_wise_feed_forward import PositionWiseFeedForward
from transformer.sub_layers.multi_head_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    """
    The EncoderLayer class defines a single layer of the transformer's encoder. It encapsulates a multi-head
    self-attention mechanism followed by the position-wise feed-forward neural network, with residual connections, layer
    normalization, and dropout applied as appropriate. Together, these components allow the encoder to capture complex
    relationships in the input data and transform them into a useful representation for downstream tasks. Typically,
    multiple such encoder layers are stacked to form the complete encoder part of a transformer model.

    The class is defined as a subclass of PyTorch's nn.Module, which means it can be used as a building block for neural
    networks in PyTorch.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        """
        Args:
            d_model (int): The input sequence length.
            num_heads (int): The number of attention heads in the multi-head attention.
            d_ff (int): The dimensionality of the inner layer in the position-wise feed-forward network.
            dropout (float): The dropout rate used for regularization for each of the layers.
        Attributes:
            self.self_attn (MultiHeadAttention): Multi-headed attention layer.
            self.feed_forward: Position-wise feed-forward network.
            self.norm1 (nn.LayerNorm) & self.norm2 (nn.LayerNorm): Layer normalization, applied to smooth the layer's
            input.
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Args:
            x: The input to the encoder layer.
            mask: Optional mask to ignore certain parts of the input sequence.
        """
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output)) # after self-attention, post-norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output)) # after feed-forward, post-norm
        return x