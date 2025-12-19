import torch
import torch.nn as nn
import math

class IMDBTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes, max_len=512):
        super().__init__()

        # 1. Token Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 2. Positional Encoding (The "Memory of Order")
        # Transformers need this to know that "dog bit man" != "man bit dog"
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # 3. The Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Final Classifier
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, src_key_padding_mask):
        # src shape: [batch_size, seq_len] (Your input_ids)
        # src_key_padding_mask shape: [batch_size, seq_len] (True where padding exists)

        # A. Embed and Add Position Info
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)

        # B. Pass through Transformer
        # We pass the mask so the attention mechanism ignores [PAD] tokens completely
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        # C. Pooling strategy: "CLS Token"
        # We take the first token (index 0) from every batch sequence
        # output shape is [batch_size, seq_len, d_model]
        cls_token_output = output[:, 0, :]

        # D. Classify
        logits = self.classifier(self.dropout(cls_token_output))
        return logits

# --- Helper Module: Positional Encoding ---
# (Standard PyTorch boilerplate you usually copy-paste)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Create a matrix of [max_len, d_model] representing positions
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]
