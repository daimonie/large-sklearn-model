import torch
import click
import os
import pickle
from data import get_data, process_data, get_tokenized
from train import train, create_transformer
from model import SmallTransformer, CustomTransformer
from data import get_titles_for_ids

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
@click.option('--model-type', type=click.Choice(['small_transformer', 'custom_transformer']), default='small_transformer')
def train(model_type):
    """Train the model"""
    file_path = get_data()  # Step 1: Get or download the dataset
    processed_data = process_data(file_path)  # Step 2: Process the data
    tokenized_data, vocab = get_tokenized(processed_data)  # Step 3: Tokenize the data

    # Step 4: Train the model
    model = train(tokenized_data, vocab, model_type=model_type)
    torch.save(model.state_dict(), f"{model_type}.pth")
    print(f"Model trained and saved to {model_type}.pth")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@cli.command()
@click.option('--model-type', type=click.Choice(['small_transformer', 'custom_transformer']), default='small_transformer')
def predict(model_type):
    """Make predictions with the trained model"""

    # Load the vocabulary and model
    # Check if cached data exists
    if os.path.exists("vocab_cache.pkl"):
        with open("vocab_cache.pkl", "rb") as f:
            vocab = pickle.load(f)
        with open("num_classes_cache.pkl", "rb") as f:
            num_classes = pickle.load(f)
    else:
        # Generate and cache if not exists
        file_path = get_data()
        processed_data = process_data(file_path)
        _, vocab = get_tokenized(processed_data)
        num_classes = len(set(label for _, label in processed_data))
        
        # Cache the results
        with open("vocab_cache.pkl", "wb") as f:
            pickle.dump(vocab, f)
        with open("num_classes_cache.pkl", "wb") as f:
            pickle.dump(num_classes, f)


    model = create_transformer(vocab, num_classes)
    print(f"Loading model from {model_type}.pth")
    model.load_state_dict(torch.load(f"{model_type}.pth"))
    model.eval()


    # Assuming your model is instantiated as `model`
    print(f"Number of trainable parameters: {count_parameters(model)}")

    # Keep predicting until user types NO
    while True:
        # Get user input
        input_text = input("Enter a passage of text (or type NO to quit): ")
        
        if input_text.upper() == "NO":
            break
            
        input_text = input_text.split()
        tokenized_input = [vocab.get(word, vocab["<UNK>"]) for word in input_text]
        
        print(tokenized_input)

        tokenized_input += [vocab["<PAD>"]] * (50 - len(tokenized_input))  # Pad to 50 tokens
        input_tensor = torch.tensor([tokenized_input])

        with torch.no_grad():
            output = model(input_tensor)
            # Get top 5 predictions
            top_probs, top_classes = torch.topk(output, k=50, dim=1)
            top_probs = torch.nn.functional.softmax(top_probs, dim=1)
            predicted_classes = top_classes[0].tolist()
            probabilities = top_probs[0].tolist()

        # Get titles for predicted book IDs
        titles = get_titles_for_ids("gutenberg_books_index.txt", predicted_classes)
        
        print("\nTop 15 predicted books:")
        for i, (title, prob) in enumerate(zip(titles, probabilities), 1):
            if title:
                print(f"{i}. {title} (Book ID {predicted_classes[i-1]}, probability: {prob:.2%})")
            else:
                print(f"{i}. Unknown Book ID {predicted_classes[i-1]} (probability: {prob:.2%})")
        print("\n")

if __name__ == '__main__':
    cli()
