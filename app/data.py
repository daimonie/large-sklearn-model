import os
import requests
from collections import Counter
from tqdm import tqdm
import pickle

def download_gutenberg_books(file_path, num_books=5, max_retries=3):
    """
    Downloads books from Project Gutenberg and saves them to individual files in a cache directory.
    """
    # Create cache directory if it doesn't exist
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate list of all book IDs up to max ID from popular books
    max_book_id = 2701  # Max ID from original popular books list
    popular_books = [(book_id, f"Book {book_id}") for book_id in range(1, max_book_id + 1)]
    
    books_to_download = popular_books[:num_books]
    
    # Create index file to maintain compatibility
    with open(file_path, "w", encoding="utf-8") as index_file:
        print(f"\nStep 1/1: Downloading {num_books} books...")
        for book_id, title in tqdm(books_to_download, desc="Books", unit="book"):
            cache_path = os.path.join(cache_dir, f"book_{book_id}.txt")
            
            # Check if already cached
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                # Download from Project Gutenberg
                url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
                try:
                    response = requests.get(url)
                    if response.status_code == 404:  # Try alternate URL format
                        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
                        response = requests.get(url)
                    text = response.text
                    
                    # Basic header stripping (simplified version)
                    start_marker = "*** START OF"
                    end_marker = "*** END OF"
                    if start_marker in text:
                        text = text.split(start_marker)[1]
                    if end_marker in text:
                        text = text.split(end_marker)[0]
                    text = text.strip()
                    
                    # Cache the book
                    with open(cache_path, "w", encoding="utf-8") as f:
                        f.write(text)
                
                except Exception as e:
                    print(f"Failed to download book {book_id}: {str(e)}")
                    continue
            
            # Write to index file to maintain compatibility with process_data()
            index_file.write(f"BOOK_ID={book_id}\n")
            index_file.write(text + "\n")
    
    print(f"\nComplete! Books saved to {cache_dir} and indexed in {file_path}")
    return file_path

def get_data():
    """
    Checks if the Gutenberg books dataset exists; downloads it if not.
    
    Returns:
        str: Path to the dataset file.
    """
    file_path = "gutenberg_books.txt"
    if not os.path.exists(file_path):
        print("Dataset not found. Downloading Gutenberg books...")
        file_path = download_gutenberg_books(file_path)
    else:
        print(f"Dataset already exists at: {file_path}")
    return file_path

def process_data(file_path, passage_length=50):
    """
    Processes the dataset into labeled passages of text.

    Args:
        file_path (str): Path to the raw dataset file.
        passage_length (int): Number of words per passage.

    Returns:
        list: A list of (passage, label) tuples.
    """
    passages = []
    current_label = None

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().splitlines()

    current_text = []
    for line in content:
        if line.startswith("BOOK_ID="):
            # Save the previous book's passages
            if current_text:
                text = " ".join(current_text)
                words = text.split()
                for i in range(0, len(words) - passage_length, passage_length):
                    passage = " ".join(words[i:i + passage_length])
                    passages.append((passage, current_label))
            # Update the current label
            current_label = line.split("=")[1]
            current_text = []  # Reset text for the new book
        else:
            current_text.append(line)

    print(f"Processed dataset into {len(passages)} passages.")
    return passages

def tokenize_and_build_vocab(dataset, pad_token="<PAD>", unk_token="<UNK>"):
    """
    Tokenizes text passages and builds a vocabulary.
    
    Args:
        dataset (list): List of (passage, label) tuples.
        pad_token (str): Token for padding.
        unk_token (str): Token for unknown words.
    
    Returns:
        list: List of tokenized passages and labels.
        dict: Word-to-index mapping (vocabulary).
    """
    # Build the vocabulary
    all_words = [word for passage, _ in dataset for word in passage.split()]
    word_counts = Counter(all_words)
    vocab = {word: idx for idx, (word, _) in enumerate(word_counts.most_common())}
    
    # Add special tokens
    vocab[pad_token] = len(vocab)
    vocab[unk_token] = len(vocab)

    # Map passages to token IDs
    tokenized_data = []
    for passage, label in dataset:
        tokenized_passage = [vocab.get(word, vocab[unk_token]) for word in passage.split()]
        tokenized_data.append((tokenized_passage, int(label)))
    
    return tokenized_data, vocab
    
def get_tokenized(passages, force_rebuild=False):
    """
    Gets tokenized data either from filesystem or by processing raw passages.
    
    Args:
        passages (list): List of (passage, label) tuples from process_data().
        force_rebuild (bool): If True, forces rebuilding tokenization even if cached.
    
    Returns:
        tuple: (tokenized_data, vocab) where tokenized_data is list of (token_ids, label)
              and vocab is the word-to-index mapping
    """
    cache_file = "tokenized_data.pkl"
    
    # Try to load cached data if it exists and force_rebuild is False
    if not force_rebuild and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                tokenized_data, vocab = pickle.load(f)
                print("Loaded tokenized data from cache")
                return tokenized_data, vocab
        except:
            print("Failed to load cached data, rebuilding...")
    
    # Build tokenized data and vocab
    tokenized_data, vocab = tokenize_and_build_vocab(passages)
    # Cache the results
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump((tokenized_data, vocab), f)
            print("Cached tokenized data")
    except Exception as e:
        print(f"Failed to cache tokenized data: {str(e)}")
    return tokenized_data, vocab
