import torch

from transformer.Transformer import Transformer

"""
initialize a Transformer model and generate random source and target sequences that can be fed into it. The chosen 
hyperparameters determine the Transformer's specific structure and properties. This setup could be part of a larger 
script where the model is trained and evaluated on actual sequence-to-sequence tasks, such as machine translation or 
text summarization.
"""
# Vocabulary sizes for source and target sequences, both set to 5000.
src_vocab_size = 5000
tgt_vocab_size = 5000
# Dimensionality of the model's embeddings, set to 512.
d_model = 512
# Number of attention heads in the multi-head attention mechanism, set to 8.
num_heads = 8
# Number of layers for both the encoder and the decoder, set to 6.
num_layers = 6
# Dimensionality of the inner layer in the feed-forward network, set to 2048.
d_ff = 2048
# Maximum sequence length for positional encoding, set to 100.
max_seq_length = 100
# Dropout rate for regularization, set to 0.1.
dropout = 0.1

# Creates an instance of the Transformer class, initializing it with the given hyperparameters. The instance will have
# the architecture and behavior defined by these hyperparameters.
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random sample data: These random sequences can be used as inputs to the transformer model, simulating a batch
# of data with 64 examples and sequences of length 100.
# Random integers between 1 and src_vocab_size, representing a batch of source sequences with shape (64, max_seq_length).
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # Batch size of 64, sequence length of 100
# Random integers between 1 and tgt_vocab_size, representing a batch of target sequences with shape (64, max_seq_length).
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # Batch size of 64, sequence length of 100