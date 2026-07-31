import torch

from training.train import transformer, src_vocab_size, max_seq_length, tgt_vocab_size, criterion

"""
This code snippet evaluates the transformer model on a randomly generated validation dataset, computes the validation 
loss, and prints it. In a real-world scenario, the random validation data should be replaced with actual validation data
from the task you are working on. The validation loss can give you an indication of how well your model is performing on
unseen data, which is a critical measure of the model's generalization ability.
"""

# Puts the transformer model in evaluation mode. This is important because it turns off certain behaviors like dropout
# that are only used during training.
transformer.load_state_dict(torch.load("checkpoints/transformer.pt"))
transformer.eval()

# Generate random sample validation data
#   Random integers between 1 and src_vocab_size, representing a batch of validation source sequences with shape (64,
#   max_seq_length).
val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length)) # (batch_size, seq_length)
#   Random integers between 1 and tgt_vocab_size, representing a batch of validation target sequences with shape (64,
#   max_seq_length).
val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length)) # (batch_size, seq_length)

# Disables gradient computation, as we don't need to compute gradients during validation. This can reduce memory
# consumption and speed up computations.
with torch.no_grad():
    # Passes the validation source data and the validation target data (excluding the last token in each sequence)
    # through the transformer.
    # Decoder input is the ground-truth target shifted right (teacher forcing),
    # the last token is dropped since it has no next-token label to predict.
    val_output = transformer(val_src_data, val_tgt_data[:, :-1])  # Exclude the last token for teacher forcing
    # Computes the loss between the model's predictions and the validation target data (excluding the first token in
    # each sequence). The loss is calculated by reshaping the data into one-dimensional tensors and using the previously
    # defined cross-entropy loss function.
    val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
    print(f"Validation loss: {val_loss.item()}")