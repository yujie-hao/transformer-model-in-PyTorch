from pathlib import Path

from transformer.transformer import Transformer

"""
<Shared configuration>
The hyperparameters live here rather than in train.py so that train.py and evaluate.py always build an
identically-shaped model. If the two ever drift apart, load_state_dict() fails with a shape mismatch --
keeping a single definition makes that impossible.
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

# Number of examples per batch of sample data.
batch_size = 64
# Number of training epochs.
num_epochs = 100

# Checkpoint location, resolved relative to the project root so the scripts work no matter which
# working directory they are launched from (PyCharm and the terminal often differ).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "transformer.pt"


def build_model():
    """
    Construct a Transformer from the hyperparameters above.

    Both train.py and evaluate.py go through this function, so the model being saved and the model
    being loaded into are guaranteed to have the same architecture.
    """
    return Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff,
                       max_seq_length, dropout)