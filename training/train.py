import torch
from torch import nn, optim

from training import config

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
# The hyperparameters live in training/config.py -- see that file for what each value means.

# Creates an instance of the Transformer class, initializing it with the given hyperparameters. The instance will have
# the architecture and behavior defined by these hyperparameters.
transformer = config.build_model()

# Generate random sample data: These random sequences can be used as inputs to the transformer model, simulating a batch
# of data with 64 examples and sequences of length 100.
# Token 0 is reserved for padding, so the random tokens start at 1.
src_data = torch.randint(1, config.src_vocab_size, (config.batch_size, config.max_seq_length))
tgt_data = torch.randint(1, config.tgt_vocab_size, (config.batch_size, config.max_seq_length))

"""
<Training the model>
This code snippet trains the transformer model on randomly generated source and target sequences for 100 epochs. It uses
the Adam optimizer and the cross-entropy loss function. The loss is printed for each epoch, allowing you to monitor the
training progress. In a real-world scenario, you would replace the random source and target sequences with actual data
from your task, such as machine translation.

NOTE: the targets here are random integers, so there is no pattern to learn. The loss will fall to roughly
ln(tgt_vocab_size) = ln(5000) ~= 8.52 -- the loss of a model guessing uniformly -- and then stop improving. That is the
expected outcome, not a bug. Run training/sanity_check_copy_task.py for a task the model can actually learn.
"""
# Defines the loss function as cross-entropy loss. The ignore_index argument is set to 0, meaning the loss will not
# consider targets with an index of 0 (typically reserved for padding tokens).
criterion = nn.CrossEntropyLoss(ignore_index=0)
# Defines the optimizer as Adam with a learning rate of 1e-4 and specific beta values.
optimizer = optim.Adam(transformer.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

# Sets the transformer model to training mode, enabling behaviors like dropout that only apply during training.
transformer.train()

for epoch in range(config.num_epochs):
    # Clears the gradients from the previous iteration.
    optimizer.zero_grad()
    # Passes the source data and the target data (excluding the last token in each sequence) through the transformer.
    # This is common in sequence-to-sequence tasks where the target is shifted by one token.
    output = transformer(src_data, tgt_data[:, :-1])
    # Computes the loss between the model's predictions and the target data (excluding the first token in each
    # sequence). The loss is calculated by reshaping the data into one-dimensional tensors and using the cross-entropy
    # loss function.
    loss = criterion(output.contiguous().view(-1, config.tgt_vocab_size),
                     tgt_data[:, 1:].contiguous().view(-1))
    # Computes the gradients of the loss with respect to the model's parameters.
    loss.backward()
    # Updates the model's parameters using the computed gradients.
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# Persists the learned weights so that evaluate.py can load them instead of re-running training.
# Creating the directory here (rather than by hand) keeps the script runnable on a fresh clone.
config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
torch.save(transformer.state_dict(), config.CHECKPOINT_PATH)
print(f"Saved checkpoint to {config.CHECKPOINT_PATH}")