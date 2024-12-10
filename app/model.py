import torch
import torch.nn as nn


class SmallTransformer(nn.Module):
    """
    A small transformer-based model for text classification.
    Predicts the book a given text passage belongs to.
    """
    def __init__(self, vocab_size, num_classes, embed_dim=128, num_heads=4, hidden_dim=256, num_layers=2, max_length=512):
        """
        Initializes the transformer model.

        Args:
            vocab_size (int): Size of the vocabulary.
            num_classes (int): Number of output classes (books).
            embed_dim (int): Dimensionality of token embeddings.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Hidden layer size in feedforward layers.
            num_layers (int): Number of transformer encoder layers.
            max_length (int): Maximum sequence length.
        """
        super(SmallTransformer, self).__init__()

        # Token embedding: Maps tokens to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding: Adds positional information to embeddings
        self.positional_encoding = self._create_positional_encoding(embed_dim, max_length)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,  # Dimension of input embeddings
            nhead=num_heads,    # Number of attention heads
            dim_feedforward=hidden_dim,  # Hidden layer size
            activation='relu'   # Activation function
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head: Maps the final encoder output to class probabilities
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # Step 1: Embed the input tokens
        embedded = self.embedding(x)  # Shape: (batch_size, seq_length, embed_dim)

        # Step 2: Add positional encoding
        embedded += self.positional_encoding[:embedded.size(1), :].to(x.device)

        # Step 3: Pass through the transformer encoder
        # Transformer expects input shape (seq_length, batch_size, embed_dim)
        encoded = self.transformer_encoder(embedded.permute(1, 0, 2))  # Shape: (seq_length, batch_size, embed_dim)

        # Step 4: Use the output from the first token ([CLS] token)
        cls_output = encoded[0, :, :]  # Shape: (batch_size, embed_dim)

        # Step 5: Classify the output
        logits = self.classifier(cls_output)  # Shape: (batch_size, num_classes)

        return logits

    @staticmethod
    def _create_positional_encoding(embed_dim, max_length):
        """
        Creates sinusoidal positional encodings for the input sequence.

        Args:
            embed_dim (int): Dimensionality of token embeddings.
            max_length (int): Maximum sequence length.
        
        Returns:
            torch.Tensor: Positional encoding tensor of shape (max_length, embed_dim).
        """
        pos = torch.arange(0, max_length).unsqueeze(1)  # Shape: (max_length, 1)
        i = torch.arange(0, embed_dim // 2).unsqueeze(0)  # Shape: (1, embed_dim // 2)
        angle_rates = 1 / (10000 ** (2 * i / embed_dim))  # Shape: (1, embed_dim // 2)

        # Compute positional encodings
        pos_encoding = torch.zeros(max_length, embed_dim)
        pos_encoding[:, 0::2] = torch.sin(pos * angle_rates)  # Apply sine to even indices
        pos_encoding[:, 1::2] = torch.cos(pos * angle_rates)  # Apply cosine to odd indices

        return pos_encoding

class CustomTransformer(nn.Module):
    """
    A custom transformer-based model for text classification.
    """
    def __init__(self, vocab_size, num_classes, embed_dim=256, num_heads=8, num_layers=4, hidden_dim=512, max_len=5000, dropout=0.1):
        super(CustomTransformer, self).__init__()
        # Embedding layer with positional encoding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)

        # Stacking transformer encoder layers
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)]
        )

        # Fully connected classification head
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x, mask=None):
        # Apply embeddings and positional encoding
        x = self.embedding(x)
        x = self.positional_encoding(x)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Pooling for classification (mean pooling)
        x = x.mean(dim=1)  # Aggregate token-level outputs

        # Final classification
        return self.fc(x)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].to(x.device)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding size must be divisible by the number of heads."
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.attention = ScaledDotProductAttention()

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.query(Q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(K).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(V).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output, attn_weights = self.attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        return self.out(attn_output)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, V), attn_weights


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))