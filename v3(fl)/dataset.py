"""
Dataset Loading and Preprocessing - Updated for Simulated FL

This module handles loading and preprocessing of datasets with performance-based allocation.
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import BertTokenizer
from datasets import load_dataset, load_from_disk
import numpy as np

def load_ag_news_dataset():
    """Load AG News dataset
    
    Returns:
        Dataset: AG News dataset
    """
    dataset = load_from_disk("../ag_news_dataset")
    return dataset

def load_yahoo_dataset():
    """Load Yahoo Answers dataset
    
    Returns:
        Dataset: Yahoo Answers dataset
    """
    dataset = load_from_disk("../yahoo_answers_topics")
    return dataset

# Initialize tokenizer
pretrain_model_path = '../bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

def news_preprocess_function(examples):
    """Preprocess function for AG News
    
    Args:
        examples (dict): Batch of examples
        
    Returns:
        dict: Tokenized examples
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

def yahoo_preprocess_function(examples):
    """Preprocess function for Yahoo Answers
    
    Args:
        examples (dict): Batch of examples
        
    Returns:
        dict: Tokenized examples
    """
    texts = [f"{title} {content}" for title, content in 
             zip(examples["question_title"], examples["question_content"])]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

class NewsDataset(Dataset):
    """PyTorch Dataset for AG News"""
    
    def __init__(self, data):
        """Initialize dataset
        
        Args:
            data: Raw dataset
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item["label"]),
        }

class YahooDataset(Dataset):
    """PyTorch Dataset for Yahoo Answers"""
    
    def __init__(self, data):
        """Initialize dataset
        
        Args:
            data: Raw dataset
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item["topic"]),
        }

def get_train_dataloaders(batch_size=8, client_id=0, performance_factor=0.5):
    """Get training DataLoader with performance-based allocation
    
    Args:
        batch_size (int): Batch size
        client_id (int): Client identifier (for logging)
        performance_factor (float): Performance factor (0-1) determining data size
        
    Returns:
        DataLoader: Training data loader
    """
    dataset = load_yahoo_dataset()
    train_data = dataset["train"]
    
    # Calculate dataset size based on performance factor
    total_size = len(train_data)
    min_size = total_size // 10  # Minimum 1% of data
    max_size = total_size // 5   # Maximum 10% of data
    dataset_size = min(max_size, min_size + int((max_size - min_size) * performance_factor))
    
    print(f"📊 Client {client_id} (performance: {performance_factor:.1f}) allocated {dataset_size} samples")
    
    # Select subset of data
    indices = np.random.choice(total_size, dataset_size, replace=False)
    train_data = train_data.select(indices).shuffle(seed=42)

    # Preprocess and create dataset
    train_data = train_data.map(yahoo_preprocess_function, batched=True)
    train_dataset = YahooDataset(train_data)

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def get_test_dataloaders(batch_size=16):
    """Get test DataLoader
    
    Args:
        batch_size (int): Batch size
        
    Returns:
        DataLoader: Test data loader
    """
    dataset = load_yahoo_dataset()
    test_data = dataset["test"]
    test_data = test_data.map(yahoo_preprocess_function, batched=True)
    test_dataset = YahooDataset(test_data)
    return DataLoader(test_dataset, batch_size=batch_size)