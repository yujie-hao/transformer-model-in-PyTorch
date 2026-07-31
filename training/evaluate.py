import torch
from torch import nn

from training import config

"""
This code snippet evaluates the transformer model on a randomly generated validation dataset, computes the validation
loss, and prints it. In a real-world scenario, the random validation data should be replaced with actual validation data
from the task you are working on. The validation loss can give you an indication of how well your model is performing on
unseen data, which is a critical measure of the model's generalization ability.

The model is built from training/config.py and its weights are loaded from the checkpoint, rather than importing the
model from train.py -- importing train.py would re-execute the entire 100-epoch training run as an import side effect.

What this measures: per-token loss when the decoder is handed the correct prefix at every position. That is directly
comparable to the training loss, which makes it useful for spotting overfitting. It does NOT measure generation quality
-- for that, the model has to consume its own predictions one token at a time (greedy or beam decoding), where a single
early mistake compounds. That is the exposure-bias problem.
"""
if not config.CHECKPOINT_PATH.exists():
    raise SystemExit(f"No checkpoint at {config.CHECKPOINT_PATH} -- run training/train.py first.")

# Builds a model with the same architecture as the trained one, then loads the saved weights into it.
transformer = config.build_model()
transformer.load_state_dict(torch.load(config.CHECKPOINT_PATH))

# The same loss function used during training, so that the two numbers can be compared directly.
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Puts the transformer model in evaluation mode. This is important because it turns off certain behaviors like dropout
# that are only used during training. Note that it does NOT disable gradient computation -- torch.no_grad() below does.
transformer.eval()

# Generate random sample validation data
#   Random integers between 1 and src_vocab_size, representing a batch of validation source sequences with shape (64,
#   max_seq_length).
val_src_data = torch.randint(1, config.src_vocab_size, (config.batch_size, config.max_seq_length))
#   Random integers between 1 and tgt_vocab_size, representing a batch of validation target sequences with shape (64,
#   max_seq_length).
val_tgt_data = torch.randint(1, config.tgt_vocab_size, (config.batch_size, config.max_seq_length))

# Disables gradient computation, as we don't need to compute gradients during validation. This can reduce memory
# consumption and speed up computations.
with torch.no_grad():
    # Passes the validation source data and the validation target data (excluding the last token in each sequence)
    # through the transformer.
    # Decoder input is the ground-truth target shifted right (teacher forcing),
    # the last token is dropped since it has no next-token label to predict.
    val_output = transformer(val_src_data, val_tgt_data[:, :-1])
    # Computes the loss between the model's predictions and the validation target data (excluding the first token in
    # each sequence). The loss is calculated by reshaping the data into one-dimensional tensors and using the previously
    # defined cross-entropy loss function.
    val_loss = criterion(val_output.contiguous().view(-1, config.tgt_vocab_size),
                         val_tgt_data[:, 1:].contiguous().view(-1))
    print(f"Validation loss: {val_loss.item()}")