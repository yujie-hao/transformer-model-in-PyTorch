# Transformer
  - A Transformer is a neural network architecture introduced in the 2017 paper ["Attention is All You Need"](https://arxiv.org/pdf/1706.03762) -- it is the foundation of virtually every modern large language model (GPT, BERT, Claude, etc.).
---
# PyTorch
  - PyTorch is an open-source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing, primarily developed by Facebook's AI Research lab (FAIR).
  - Tutorial: https://docs.pytorch.org/tutorials/index.html
---
# Transformer Architecture
## < Architecture >
- The Transformer architecture is based on a self-attention mechanism that allows the model to weigh the importance of different words in a sentence when making predictions. It consists of an encoder and a decoder, each made up of multiple layers of self-attention and feed-forward neural networks.
![transformer_arch.png](assets/transformer_arch.png)
## < Transformer Model >
- The Transformer class brings together the various components of a Transformer model, including the embeddings, positional encoding, encoder layers, and decoder layers. It provides a convenient interface for training and inference, encapsulating the complexities of multi-head attention, feed-forward networks, and layer normalization. This implementation follows the standard Transformer architecture, making it suitable for sequence-to-sequence tasks like machine translation, text summarization, etc. Including masking ensures that the model adheres to the causal dependencies within sequences, ignoring padding tokens and preventing information leakage from future tokens. These sequential steps empower the Transformer model to efficiently process input sequences and produce corresponding output sequences.
## < Layers >
- [ Encoder Layer ] </br>
  The EncoderLayer class defines a single layer of the transformer's encoder. It encapsulates a multi-head self-attention mechanism followed by the position-wise feed-forward neural network, with residual connections, layer normalization, and dropout applied as appropriate. Together, these components allow the encoder to capture complex relationships in the input data and transform them into a useful representation for downstream tasks. Typically, multiple such encoder layers are stacked to form the complete encoder part of a transformer model.
- [ Decoder Layer ] </br>
The DecoderLayer class defines a single layer of the transformer's decoder. It consists of a multi-head self-attention mechanism, a multi-head cross-attention mechanism (that attends to the encoder's output), a position-wise feed-forward neural network, and the corresponding residual connections, layer normalization, and dropout layers. This combination enables the decoder to generate meaningful outputs based on the encoder's representations, taking into account both the target sequence and the source sequence. As with the encoder, multiple decoder layers are typically stacked to form the complete decoder part of a transformer model.
## < Sub layers >
### [ Multi-Head Attention ]
- Mechanism to focus on different parts of the input. Captures dependencies across different positions in the sequence
- ![attention.png](assets/attention.png)
- <em>[learn more](https://campus.datacamp.com/courses/large-language-models-llms-concepts/training-methodology-and-techniques?ex=8#)</em>
### [ Position-wise Feed-Forward Networks (FFN) ]
- FFN is the neural network part that processes each token independently after attention.
- It consists of two linear transformations with a ReLU activation in between.
  - The 1st linear layer expands the dimensionality of the input (512 → 2048).
  - The 2nd linear layer projects it back to the original dimension (2048 → 512).
  - Formula: FFN(x) = max(0,xW1 + b1)W2 + b2
### [ Positional Encoding ]
- The PositionalEncoding class adds information about the position of tokens within the sequence. Since the 
transformer model lacks inherent knowledge of the order of tokens (due to its self-attention mechanism), this 
class helps the model to consider the position of tokens in the sequence. The sinusoidal functions used are 
chosen to allow the model to easily learn to attend to relative positions, as they produce a unique and smooth 
encoding for each position in the sequence.
---
# Processing steps
## Encoder layer
1. Self-attention: The input x is passed through the multi-head self-attention mechanism.
2. Add and normalize (after attention): The attention output is added to the original input (residual connection),
  followed by dropout and normalization using norm1.
3. Feed-forward network: The output from the previous step is passed through the position-wise feed-forward
  network.
4. Add and normalize (after feed-forward): Similar to step 2, the feed-forward output is added to the input of
  this stage (residual connection), followed by dropout and normalization using norm2.
5. Output: The processed tensor is returned as the output of the encoder layer.

### [ What LayerNorm actually computes ]
For each token vector independently, it normalizes across the d_model features — subtract that vector's mean,
divide by its standard deviation, then apply a learned scale γ and shift β:
for each token:  y = γ · (x - mean(x)) / sqrt(var(x) + ε) + β
With d_model=512, each nn.LayerNorm(512) owns 1024 learnable parameters (512 for γ, 512 for β).
Shape is unchanged: (batch, seq_len, 512) in, same out.

### [ Why two separate instances rather than one reused ]
Because of those learned γ/β — the post-attention distribution and the post-FFN distribution are different, so
each needs its own. Writing self.norm1(...) twice would force them to share parameters, which is a real (and
subtle) modeling bug.

### [ Norm improvement ]
- Post-norm: This is post-norm (norm applied after the residual add), matching the 2017 paper.
- Pre-norm: Most modern implementations use pre-norm — x = x + self.dropout(self.self_attn(self.norm1(x), ...)) —
  which trains more stably at depth and typically doesn't need learning-rate warmup.
## Decoder layer
1. Self-attention on target sequence: The input x is processed through a self-attention mechanism.
2. Add and normalize (after self-attention): The output from self-attention is added to the original x, followed
by dropout and normalization using norm1.
3. Cross-attention with encoder output: The normalized output from the previous step is processed through a
cross-attention mechanism that attends to the encoder's output enc_output.
4. Add and normalize (after cross-attention): The output from cross-attention is added to the input of this stage,
followed by dropout and normalization using norm2.
5. Feed-forward network: The output from the previous step is passed through the feed-forward network.
6. Add and normalize (after feed-forward): The feed-forward output is added to the input of this stage, followed
by dropout and normalization using norm3.
7. Output: The processed tensor is returned as the output of the decoder layer.
---
# Execution
## < Setup >
- The project needs only PyTorch:
  ```bash
  python -m venv .venv
  .venv/bin/pip install torch
  ```
## < Usage >
- The scripts import across packages (`transformer.*`, `training.*`), so run them from the repository root with
  that root on `PYTHONPATH`:
  ```bash
  PYTHONPATH=. .venv/bin/python training/sanity_check_copy_task.py   # 1. verify the architecture
  PYTHONPATH=. .venv/bin/python training/train.py                    # 2. train, writes the checkpoint
  PYTHONPATH=. .venv/bin/python training/evaluate.py                 # 3. load the checkpoint and score it
  ```
- In PyCharm no `PYTHONPATH` is needed -- the repository root is already the content root, so the scripts can be
  run directly from the editor.
## < Scripts >
- `training/config.py` -- the shared hyperparameters plus `build_model()`.
- `training/sanity_check_copy_task.py` -- trains a small model on a learnable toy task (copy the source sequence)
  to confirm the masking and the target shift are correct. Runs on CPU in seconds. Start here.
- `training/train.py` -- trains the full 51.8M-parameter model on random sample data for 100 epochs, then saves
  the weights to `checkpoints/transformer.pt`.
- `training/evaluate.py` -- rebuilds the model, loads the checkpoint, and reports the teacher-forced validation
  loss on a fresh batch. Requires `train.py` to have been run first.
---
# Summary
## < Sanity check >
```
Step:    1, Loss: 4.0926
Step:  400, Loss: 0.0002

Final loss: 0.0002   teacher-forced token accuracy: 100.0%
PASS -- the architecture learns.
```
- On the copy task the model reaches 100% token accuracy, which confirms that `generate_mask()` and the
  `tgt[:, :-1]` / `tgt[:, 1:]` shift are implemented correctly.
## < Training and Evaluation >
```
training/train.py     ->  Epoch: 100, Loss: 2.727
training/evaluate.py  ->  Validation loss: 8.821
```
## < Interpretation >
- `src_data` and `tgt_data` are random integers generated **once** and reused every epoch, so the training loss
  of ~2.73 is the model memorizing 64 specific sequences rather than learning a pattern.
- Evaluation draws a **fresh** random batch, so the validation loss of ~8.82 is marginally worse than uniform
  guessing across the 5000-token vocabulary (ln 5000 ≈ 8.52).
- The gap between the two is textbook overfitting, and it is the expected outcome here: random data contains no
  pattern to generalize from. The pipeline is working; the data is the placeholder.
- Note: Turning this into a model that can actually translate or summarize needs three further pieces: a real 
  dataset (e.g. Multi30k for translation), a tokenizer to map text to token IDs, and an autoregressive decoding loop
  (greedy or beam search) -- `forward()` is teacher-forced and consumes the whole target at once, so it cannot
  generate text on its own.
---
# Reference
  - https://arxiv.org/pdf/1706.03762
  - https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
  - https://www.geeksforgeeks.org/deep-learning/transformer-using-pytorch/