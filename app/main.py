import torch
import click
from data import get_data, process_data, get_tokenized
from train import train
from model import SmallTransformer 

@click.group()
def cli():
    """CLI for managing ML workflow"""
    print("Is CUDA available:", torch.cuda.is_available())
    print("CUDA Device Name:", torch.cuda.get_device_name(0))

@cli.command()
def generate():
    """Download training data"""
    file_path = get_data()

@cli.command()
def preprocess():
    """Download training data"""
    file_path = get_data()
    processed_data = process_data(file_path)
    tokenized_data, vocabulary = get_tokenized(processed_data)

@cli.command()
def train():
    """Train the model"""
    file_path = get_data()  # Step 1: Get or download the dataset
    processed_data = process_data(file_path)  # Step 2: Process the data
    tokenized_data, vocab = get_tokenized(processed_data)  # Step 3: Tokenize the data

    # Step 4: Train the model
    from train import train
    model = train(tokenized_data, vocab)
    torch.save(model.state_dict(), "small_transformer.pth")
    print("Model trained and saved to small_transformer.pth")

@cli.command()
def predict():
    """Make predictions with the trained model"""

    # Load the vocabulary and model
    file_path = get_data()
    processed_data = process_data(file_path)
    _, vocab = get_tokenized(processed_data)

    num_classes = len(set(label for _, label in processed_data))
    model = SmallTransformer(len(vocab), num_classes)
    model.load_state_dict(torch.load("small_transformer.pth"))
    model.eval()

    # Get user input and predict
    input_text = input("Enter a passage of text: ").split()
    tokenized_input = [vocab.get(word, vocab["<UNK>"]) for word in input_text]
    tokenized_input += [vocab["<PAD>"]] * (50 - len(tokenized_input))  # Pad to 50 tokens
    input_tensor = torch.tensor([tokenized_input])

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    print(f"Predicted Book ID: {predicted_class}")

if __name__ == '__main__':
    cli()
