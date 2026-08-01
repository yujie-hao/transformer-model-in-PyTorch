import torch
from torch import nn

from transformer.layers.decoder_layer import DecoderLayer
from transformer.layers.encoder_layer import EncoderLayer
from transformer.sub_layers.positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        """
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Transformer model's embedding dimension
            num_heads: Number of attention heads in the multi-head attention mechanism
            num_layers: Number of layers for both the encoder and the decoder
            d_ff: Dimensionality of the inner layer in the feed-forward network
            max_seq_length: Maximum sequence length for positional encoding
            dropout: Dropout rate for regularization
        Attributes:
            self.encoder_embedding: Transformer embedding layer for the source sequence
            self.decoder_embedding: Transformer embedding layer for the target sequence
            self.positional_encoding: Transformer positional encoding layer
            self.encoder_layers: A stack of encoder layers
            self.decoder_layers: A stack of decoder layers
            self.fc: Final fully connected (linear) layer mapping to target vocabulary size
            self.dropout: Dropout layer
        """
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        """
        Create masks for the source and target sequences, ensuring that padding tokens are ignored and that future
        tokens are not visible during training for the target sequence.
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        """
        Defines the forward pass for the Transformer, taking source and target sequences and producing the output
        predictions.
          1. Input embedding and positional encoding: The source and target sequences are first embedded using their
          respective embedding layers and then added to their positional encodings.
          2. Encoder layers: The source sequence is passed through the encoder layers, with the final encoder output
          representing the processed source sequence.
          3. Decoder layers: The target sequence and the encoder's output are passed through the decoder layers,
          resulting in the decoder's output.
          4. Final linear layer: The decoder's output is mapped to the target vocabulary size using a fully connected
          (linear) layer.

        Output: The final output is a tensor representing the model's predictions for the target sequence.
        """
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output =  src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
