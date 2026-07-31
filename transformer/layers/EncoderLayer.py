from torch import nn

from transformer.sub_layers.PositionWiseFeedForward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    """
    The EncoderLayer class defines a single layer of the transformer's encoder. It encapsulates a multi-head
    self-attention mechanism followed by the position-wise feed-forward neural network, with residual connections, layer
    normalization, and dropout applied as appropriate. Together, these components allow the encoder to capture complex
    relationships in the input data and transform them into a useful representation for downstream tasks. Typically,
    multiple such encoder layers are stacked to form the complete encoder part of a transformer model.

    The class is defined as a subclass of PyTorch's nn.Module, which means it can be used as a building block for neural
    networks in PyTorch.

    <Processing steps>
      1. Self-attention: The input x is passed through the multi-head self-attention mechanism.
      2. Add and normalize (after attention): The attention output is added to the original input (residual connection),
        followed by dropout and normalization using norm1.
      3. Feed-forward network: The output from the previous step is passed through the position-wise feed-forward
        network.
      4. Add and normalize (after feed-forward): Similar to step 2, the feed-forward output is added to the input of
        this stage (residual connection), followed by dropout and normalization using norm2.
      5. Output: The processed tensor is returned as the output of the encoder layer.

    <What LayerNorm actually computes>
      For each token vector independently, it normalizes across the d_model features — subtract that vector's mean,
      divide by its standard deviation, then apply a learned scale γ and shift β:
      for each token:  y = γ · (x - mean(x)) / sqrt(var(x) + ε) + β
      With d_model=512, each nn.LayerNorm(512) owns 1024 learnable parameters (512 for γ, 512 for β).
      Shape is unchanged: (batch, seq_len, 512) in, same out.

    <Why two separate instances rather than one reused>
      Because of those learned γ/β — the post-attention distribution and the post-FFN distribution are different, so
      each needs its own. Writing self.norm1(...) twice would force them to share parameters, which is a real (and
      subtle) modeling bug.

    <Norm improvement>
      - Post-norm: This is post-norm (norm applied after the residual add), matching the 2017 paper.
      - Pre-norm: Most modern implementations use pre-norm — x = x + self.dropout(self.self_attn(self.norm1(x), ...)) —
        which trains more stably at depth and typically doesn't need learning-rate warmup.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        """
        Args:
            d_model (int): The input sequence length.
            num_heads (int): The number of attention heads in the multi-head attention.
            d_ff (int): The dimensionality of the inner layer in the position-wise feed-forward network.
            dropout (float): The dropout rate used for regularization for each of the layers.
        Attributes:
            self.self_attn (nn.MultiheadAttention): Multi-headed attention layer.
            self.feed_forward: Position-wise feed-forward network.
            self.norm1 (nn.LayerNorm) & self.norm2 (nn.LayerNorm): Layer normalization, applied to smooth the layer's
            input.
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
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