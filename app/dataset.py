import torch
from torch.utils.data import Dataset

def create_dataset(tokenized_data, vocab, seq_length=50):
    """
    Creates a GutenbergDataset from passages.
    
    Args:
        passages (list): List of (passage, label) tuples from process_data().
        seq_length (int): Fixed length for sequences.
        force_rebuild (bool): Whether to force rebuild tokenization.
    
    Returns:
        GutenbergDataset: Dataset ready for training.
        dict: The vocabulary mapping.
    """ 
    
    # Create dataset with padding token index from vocabulary
    pad_idx = vocab.get("<PAD>", 0)
    dataset = GutenbergDataset(tokenized_data, seq_length=seq_length, pad_idx=pad_idx)
    
    return dataset, vocab


class GutenbergDataset(Dataset):
    def __init__(self, tokenized_data, seq_length=50, pad_idx=0):
        """
        Initializes the dataset.
        
        Args:
            tokenized_data (list): List of (tokenized_passage, label) tuples.
            seq_length (int): Fixed length for sequences.
            pad_idx (int): Index for padding tokens.
        """
        self.data = tokenized_data
        self.seq_length = seq_length
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, label = self.data[idx]

        # Pad or truncate tokens
        tokens = tokens[:self.seq_length]
        tokens += [self.pad_idx] * (self.seq_length - len(tokens))

        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)
