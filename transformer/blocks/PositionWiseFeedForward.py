from torch import nn

# PositionWiseFeedForward is a subclass of PyTorch's nn.Module, it will inherit all functionalities required to work
# with neural network layers.
class PositionWiseFeedForward(nn.Module):
    """ FFN is the neural network part that processes each token independently after attention.
    It consists of two linear transformations with a ReLU activation in between. The first linear layer expands the
    dimensionality of the input, and the second linear layer projects it back to the original dimension.

    FFN(x) = max(0,xW1 + b1)W2 + b2

    <Key purpose>
    1. Adds Non-Linearity: Self-attention is primarily linear. FFN introduces non-linearity via the ReLU activation,
    to learn complex patterns.
    2. Per-Position Feature Transformation: Attention mixes info across positions, while FFN transforms features at
    each position independently, allowing for richer representations.
        [Attention]: "Which other tokens should this token pay attention to?"
        [FFN]:       "Given the information collected by attention, how should this token transform its own
        representation?"
    3. Dimensionality Expansion: The hidden layer is typically 4× the model dimension (e.g., 512 → 2048 → 512).
    This gives the model more parameters to store learned patterns and transformations.

    <What "position-wise" means>
    The same feed-forward network is applied independently to every position (token) in the sequence — identical
    weights, no mixing between positions:
    Input: [token1_vector, token2_vector, token3_vector, ..., token_n_vector]
    Position-Wise Feed-Forward applies the SAME network to each:
      token1_vector  → [Linear → ReLU → Linear] → transformed_token1_vector
      token2_vector  → [Linear → ReLU → Linear] → transformed_token2_vector
      token3_vector  → [Linear → ReLU → Linear] → transformed_token3_vector
      ...
      token_n_vector → [Linear → ReLU → Linear] → transformed_token_n_vector
    Each transformation is INDEPENDENT (no information flows between positions)
    """
    def __init__(self, d_model, d_ff):
        """
        Args:
            d_model: Dimensionality of the token embeddings as model's input & output
            d_ff: Dimensionality of the hidden (inner) layer in the FFN

        Attributes:
            fc1: Fully connected (linear) layer expanding d_model -> d_ff
            fc2: Fully connected (linear) layer contracting d_ff -> d_model
            relu: Rectified Linear Unit activation function, introduces non-linearity between the two linear layers
        """
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff) # 512 -> 2048 (expand)
        self.fc2 = nn.Linear(d_ff, d_model) # 2048 -> 512 (contract)
        self.relu = nn.ReLU()

    def forward(self, x):
        """ Overrides the function in nn.Module - defines what the layer actually computes.
        computation FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
        execution:
          ffn = PositionWiseFeedForward(d_model=512, d_ff=2048)
          out = ffn(x) --> nn.Module.__call__ dispatches to forward
        Args:
            x: [token1_vector, token2_vector, token3_vector, ...] -> The input to the FFN, typically the output from
            the attention mechanism.
        Attributes:
            self.fc1(x): The input is first passed through the first linear layer (fc1).
            self.relu(...): The output of fc1 is then passed through a ReLU activation function. ReLU replaces all
            negative values with zeros, introducing non-linearity into the layers.
            self.fc2(...): The activated output is then passed through the second linear layer (fc2), producing the
            final output.
        Returns:
            The transformed output after applying the two linear layers and the ReLU activation in between.
        """
        return self.fc2(self.relu(self.fc1(x)))