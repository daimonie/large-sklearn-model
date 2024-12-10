import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import SmallTransformer, CustomTransformer  # Import your transformer model
from dataset import create_dataset  # Import the dataset creation function

def create_transformer(vocab, num_classes, model_type="small_transformer"):

    if model_type == "small_transformer":
        return SmallTransformer(
            vocab_size=len(vocab),
            num_classes=num_classes,
            embed_dim=256,  # Larger embedding dimension
            num_heads=8,    # More attention heads
            num_layers=4    # More transformer layers
        )
    elif model_type == "custom_transformer":
        return CustomTransformer(
            vocab_size=len(vocab),
            num_classes=num_classes,
            embed_dim=256,
            num_heads=8,
            num_layers=4
        )

def train(tokenized_data, vocab, num_epochs=30, batch_size=32, seq_length=50, model_type="small_transformer"):
    """
    Trains the SmallTransformer model on the tokenized Gutenberg dataset.
    
    Args:
        tokenized_data (list): Tokenized passages and labels.
        vocab (dict): Vocabulary mapping words to indices.
        num_epochs (int): Number of training epochs.
        batch_size (int): Size of each training batch.
        seq_length (int): Maximum sequence length for input data.
    """
    # Create the dataset and DataLoader
    dataset, _ = create_dataset(tokenized_data, vocab, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    num_classes = len(set(label for _, label in tokenized_data))
    print(f"Number of classes: {num_classes}")
    vocab_size = len(vocab)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model = create_transformer(vocab, num_classes, model_type=model_type).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001 * batch_size / 32)

    for _, label in tokenized_data:
        assert 0 <= label < num_classes, f"Invalid label found: {label}"

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training epochs", unit="epoch"):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, labels) in tqdm(enumerate(dataloader), desc=f"Epoch {epoch+1}", total=len(dataloader), unit="batch"):
            # print(f"Batch {batch_idx}: Labels: {labels}")
            inputs, labels = inputs.to(device), labels.to(device) 

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

    # Save the trained model
    return model
