import os
import requests
from collections import Counter
from tqdm import tqdm
import pickle
import random

def download_gutenberg_books(file_path, max_retries=3):
    """
    Downloads books from Project Gutenberg and saves them to individual files in a cache directory.
    """
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate random sample of Gutenberg book IDs and create our own sequential IDs
    max_gutenberg_id = 5000
    num_books = 5000

    gutenberg_ids = list(range(1, max_gutenberg_id + 1))
    if num_books < max_gutenberg_id:
        gutenberg_ids = set()
        while len(gutenberg_ids) < num_books:
            gutenberg_ids.add(random.randint(1, max_gutenberg_id))
        gutenberg_ids = list(gutenberg_ids)
    

    # Create a mapping dictionary to store sequential ID to Gutenberg ID mapping
    id_mapping = {} 
    
    index_content = []  # New: Store index content in memory
    successful_downloads = 0  # New: Track successful downloads

    skip_book_if_contains = [
        "<h1>Forbidden</h1>",
        "<title>403 Forbidden</title>",
        "This file is available in several formats, .avi, .mpeg as follows",
        "The pages are contained within the accompanying .zip file.",
        "Please see the corresponding RTF file for this eBook.",
        "THE PROJECT GUTENBERG EBOOK REPERTORY OF THE COMEDIE HUMAINE, PART 2 ***" # unsure why this book doesn't wokr
    ]
    
    print(f"\nStep 1/1: Downloading {num_books} books...")
    for gutenberg_id in tqdm(gutenberg_ids, desc="Books", unit="book"):
        cache_path = os.path.join(cache_dir, f"book_{gutenberg_id}.txt")
        
        try:
            # Check if already cached
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                # Download from Project Gutenberg using gutenberg_id
                url = f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt" 
                response = requests.get(url)
                if response.status_code == 404:  # Try alternate URL format
                    url = f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"
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
                text = '\n'.join(line for line in text.splitlines() if line.strip())
                
                # Cache the book
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)

            # Only add to id_mapping and index_content if text is substantial enough
            if len(text) >= 150 and not any(skip_word in text for skip_word in skip_book_if_contains):
                our_id = successful_downloads  # Use successful_downloads instead of len(id_mapping)
                id_mapping[our_id] = gutenberg_id
                
                # Move these lines INSIDE the if condition to ensure consistency
                index_content.append(f"BOOK_ID={our_id}")
                index_content.append(text)
                successful_downloads += 1
            else:
                # Don't add anything to index_content if the book is invalid
                # Still cache invalid books so they're skipped next time
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write("")
                continue  # Skip this book if text is too short
            
            
        except Exception as e:
            print(f"Failed to download book {gutenberg_id}: {str(e)}")
            continue
    
    # New: Write index file only after all downloads are complete
    # New: Write index file only after all downloads are complete
    index_file_path = file_path.replace(".", "_index.")
    with open(index_file_path, "w", encoding="utf-8") as index_file:
        index_file.write("\n".join([f"{our_id}={gutenberg_id}" for our_id, gutenberg_id in id_mapping.items()]))
    
    with open(file_path, "w", encoding="utf-8") as index_file:
        index_file.write("\n".join(index_content))
    
    print(f"\nComplete! {successful_downloads} books saved to {cache_dir} and indexed in {file_path}")
    return file_path
def get_book_title(gutenberg_id):
    """
    Retrieves the title of a book from Project Gutenberg using its ID.
    
    Args:
        gutenberg_id (str): The Gutenberg ID of the book
        
    Returns:
        str: The title of the book, or None if not found
    """
    try:
        # Try the main URL format first
        url = f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt"
        response = requests.get(url)
        
        # If 404, try alternate URL format
        if response.status_code == 404:
            url = f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"
            response = requests.get(url)
            
        if response.status_code != 200:
            return None
            
        text = response.text
        
        # Look for title in the header section
        title = None
        for line in text.splitlines()[:100]:  # Check first 100 lines
            if "Title:" in line:
                title = line.split("Title:")[1].strip()
                break
                
        return title
        
    except Exception as e:
        print(f"Error retrieving title for book {gutenberg_id}: {str(e)}")
        return None

def get_titles_for_ids(index_file_path, our_ids):
    """
    Gets titles for specified book IDs using the index file mapping.
    
    Args:
        index_file_path (str): Path to the index file containing ID mappings
        our_ids (list): List of our internal book IDs
        
    Returns:
        list: List of titles in the same order as our_ids (None for any failures)
    """
    # Read the ID mapping from index file
    id_to_gutenberg = {}
    try:
        with open(index_file_path, "r", encoding="utf-8") as f:
            for line in f:
                our_id, gutenberg_id = line.strip().split("=")
                id_to_gutenberg[int(our_id)] = gutenberg_id
    except Exception as e:
        print(f"Error reading index file: {str(e)}")
        return [None] * len(our_ids)
    
    # Get titles for requested IDs
    titles = []
    for our_id in our_ids:
        if our_id in id_to_gutenberg:
            title = get_book_title(id_to_gutenberg[our_id])
            titles.append(title)
        else:
            titles.append(None)
            
    return titles


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
    passages = []
    current_label = None

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().splitlines()
    book_ids = [line.split('=')[1].strip() for line in content if line.startswith("BOOK_ID=")]
    unique_book_ids = set(book_ids)
    print(f"Unique book IDs found: {len(unique_book_ids)}")

    # Track which books have passages
    books_with_passages = set()
    
    current_text = []
    for line in tqdm(content, desc="Processing text", unit="line"):
        if line.startswith("BOOK_ID="):
            # Process previous book if it exists and has content
            if current_text and current_label is not None:
                text = " ".join(current_text)
                words = text.split()
                if len(words) >= passage_length:  # Only process if enough words
                    books_with_passages.add(current_label)
                    # Only take first 50 passages per book
                    for i in range(0, min(50 * passage_length, len(words) - passage_length), passage_length):
                        passage = " ".join(words[i:i + passage_length])
                        passages.append((passage, int(current_label)))
                else:
                    print(f"Skipping book with insufficient words: {current_label}")
                    print(current_text)
            current_label = line.split("=")[1]
            current_text = []
        else:
            # Skip empty lines and only add non-empty content
            if line.strip():
                current_text.append(line.strip())
            
    # Process the last book 
    if current_text and current_label is not None:
        text = " ".join(current_text)
        words = text.split()
        if len(words) >= passage_length:
            books_with_passages.add(current_label)
            for i in range(0, len(words) - passage_length, passage_length):
                passage = " ".join(words[i:i + passage_length])
                passages.append((passage, int(current_label)))

    processed_labels = set(label for _, label in passages)
    print(f"Processed {len(passages)} passages with {len(processed_labels)} unique labels.")
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
    for passage, label in tqdm(dataset, desc="Tokenizing passages", unit="passage"):
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
