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
