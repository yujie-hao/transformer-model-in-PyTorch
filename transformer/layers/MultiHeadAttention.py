import math

import torch
import torch.nn as nn

# The MultiHeadAttention class is a subclass of PyTorch's nn.Module (Base class for all neural network modules)
class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention class encapsulates the multi-head attention mechanism commonly used in transformer models.
    It takes care of splitting the input into multiple attention heads, applying attention to each head, and then
    combining the results. By doing so, the model can capture various relationships in the input data at different
    scales, improving the expressive ability of the model.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        self.d_model = d_model # Model's embedding dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, value

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """ Calculate attention scores
        Calculate attention scores, applies a mask if provided and returns the final output
        """
        # Here, the attention scores are calculated by taking the dot product of queries (Q) and keys (K), and then
        # scaling by the square root of the key dimension (d_k).
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # The attention scores are passed through a softmax function to convert them into probabilities that sum to 1.
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        

    def split_heads(self, x):
        # Reshapes the input x into the shape (batch_size, num_heads, seq_length, d_k). It enables the model to
        # process multiple attention heads concurrently, allowing for parallel computation.
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # After applying attention to each head separately, this method combines the results back into a single tensor
        # of shape (batch_size, seq_length, d_model). This prepares the result for further processing.
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        The forward method is where the actual computation happens:
        1. <Apply linear transformations>: The queries (Q), keys (K), and values (V) are first passed through linear
        transformations using the weights defined in the initialization.
        2. <Split heads>: The transformed Q, K, V are split into multiple heads using the split_heads method.
        3. <Apply scaled dot-product attention>: The scaled_dot_product_attention method is called on the split heads.
        4. <Combine heads>: The results from each head are combined back into a single tensor using the combine_heads
        method.
        5. <Apply output transformation>: Finally, the combined tensor is passed through an output linear
        transformation.
        """

        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output