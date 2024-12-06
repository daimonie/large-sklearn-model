import torch
import click
from data import create_data

@click.group()
def cli():
    """CLI for managing ML workflow"""
    print("Is CUDA available:", torch.cuda.is_available())
    print("CUDA Device Name:", torch.cuda.get_device_name(0))

@cli.command()
def generate():
    """Generate training data"""
    create_data()

@cli.command() 
def train():
    """Train the model"""
    raise NotImplementedError("Training not implemented yet")

@cli.command()
def predict():
    """Make predictions with trained model"""
    raise NotImplementedError("Prediction not implemented yet")

if __name__ == '__main__':
    cli()
