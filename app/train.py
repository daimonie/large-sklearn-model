import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SmallTransformer  # Import your transformer model
from dataset import create_dataset  # Import the dataset creation function

def train(tokenized_data, vocab, num_epochs=10, batch_size=32, seq_length=50):
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
    model = SmallTransformer(vocab_size, num_classes).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _, label in tokenized_data:
        assert 0 <= label < num_classes, f"Invalid label found: {label}"

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            print(f"Batch {batch_idx}: Labels: {labels}")
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
