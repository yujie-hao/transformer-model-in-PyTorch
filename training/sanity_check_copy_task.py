import torch
from torch import nn, optim

from transformer.Transformer import Transformer

"""
<Architecture sanity check: the copy task>
train.py trains on random targets, which are unlearnable by construction -- the loss settles at
ln(tgt_vocab_size) ~= 8.52 no matter how long it runs. That means train.py cannot tell you whether the
architecture itself is correct; a model with a broken mask and a model with a perfect one both plateau there.

This script gives the model a task that IS learnable, but trivially so: copy the source sequence into the target.
A correct transformer overfits a single fixed batch of this within a few hundred steps and drives the loss to
near zero. If the loss plateaus instead, there is a real bug -- most likely in the masking (generate_mask) or in
the target shift -- and it is far cheaper to find it here than after wiring up a real dataset and tokenizer.

The model is deliberately small (2 layers, d_model=128, sequences of length 10) so this runs on CPU in seconds
rather than the minutes the full 512-dimensional, 6-layer model takes.
"""
# A fixed seed keeps this reproducible: the same run should give the same numbers every time.
torch.manual_seed(0)

# Small model + short sequences: this is a debugging harness, not a training run.
vocab_size = 50
d_model = 128
num_heads = 4
num_layers = 2
d_ff = 512
seq_length = 10
batch_size = 32
# Dropout is switched off on purpose. We WANT the model to overfit this one batch, and dropout only adds noise
# that slows that down.
dropout = 0.0
# A higher learning rate than train.py uses -- the task is easy and we want convergence in seconds.
learning_rate = 1e-3
num_steps = 400

model = Transformer(vocab_size, vocab_size, d_model, num_heads, num_layers, d_ff, seq_length, dropout)

# The task: the target is an exact copy of the source. Token 0 is reserved for padding, so tokens start at 1.
src_data = torch.randint(1, vocab_size, (batch_size, seq_length))
tgt_data = src_data.clone()

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

model.train()
for step in range(1, num_steps + 1):
    optimizer.zero_grad()
    output = model(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, vocab_size),
                     tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    if step == 1 or step % 50 == 0:
        print(f"Step: {step:4d}, Loss: {loss.item():.4f}")

# Checks that the model can actually reproduce the sequences it was just trained on. Teacher-forced accuracy:
# given the correct prefix at every position, how often is the next-token prediction right?
model.eval()
with torch.no_grad():
    output = model(src_data, tgt_data[:, :-1])
    predicted = output.argmax(dim=-1)
    accuracy = (predicted == tgt_data[:, 1:]).float().mean().item()

print(f"\nFinal loss: {loss.item():.4f}   teacher-forced token accuracy: {accuracy:.1%}")
if accuracy > 0.95:
    print("PASS -- the architecture learns. The plateau in train.py is the random data, not a bug.")
else:
    print("FAIL -- did not converge. Check generate_mask() and the tgt[:, :-1] / tgt[:, 1:] shift.")