from datasets import load_dataset
import os

def download_wikipedia_dataset(file_path):
    # Download a subset of the Wikipedia dataset (e.g., English articles)
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:1%]")  # Adjust percentage as needed
    
    # Save the dataset locally
    with open(file_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(item['text'] + "\n")
    
    print("Wikipedia dataset downloaded and saved to wikipedia_text.txt")
    return file_path

def get_data():
    file_path = "wikipedia_text.txt"
    if not os.path.exists(file_path):
        file_path = download_wikipedia_dataset(file_path)
    print(f"Dataset saved at: {file_path}")
    return file_path
