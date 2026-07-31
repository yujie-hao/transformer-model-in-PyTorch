import torch
from torch import nn, optim

from transformer.Transformer import Transformer

"""
<Sample data preparation>
For illustrative purposes, a dummy dataset will be crafted in this example. However, in a practical scenario, a more 
substantial dataset would be employed, and the process would involve text preprocessing along with the creation of 
vocabulary mappings for both the source and target languages.

The code snippet demonstrates how to initialize a Transformer model and generate random source and target sequences that
can be fed into it. The chosen hyperparameters determine the Transformer's specific structure and properties. This setup
could be part of a larger script where the model is trained and evaluated on actual sequence-to-sequence tasks, such as 
machine translation or text summarization.
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

"""
<Training the model>
This code snippet trains the transformer model on randomly generated source and target sequences for 100 epochs. It uses
the Adam optimizer and the cross-entropy loss function. The loss is printed for each epoch, allowing you to monitor the
training progress. In a real-world scenario, you would replace the random source and target sequences with actual data
from your task, such as machine translation.
"""
# Defines the loss function as cross-entropy loss. The ignore_index argument is set to 0, meaning the loss will not
# consider targets with an index of 0 (typically reserved for padding tokens).
criterion = nn.CrossEntropyLoss(ignore_index=0)
# Defines the optimizer as Adam with a learning rate of 0.0001 and specific beta values.
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Sets the transformer model to training mode, enabling behaviors like dropout that only apply during training.
transformer.train()

for epoch in range(100):
    # Clears the gradients from the previous iteration.
    optimizer.zero_grad()
    # Passes the source data and the target data (excluding the last token in each sequence) through the transformer.
    # This is common in sequence-to-sequence tasks where the target is shifted by one token.
    output = transformer(src_data, tgt_data[:, :-1])
    # Computes the loss between the model's predictions and the target data (excluding the first token in each
    # sequence). The loss is calculated by reshaping the data into one-dimensional tensors and using the cross-entropy
    # loss function.
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    # Computes the gradients of the loss with respect to the model's parameters.
    loss.backward()
    # Updates the model's parameters using the computed gradients.
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")