"""
Dataset Loading and Preprocessing - Updated for Simulated FL

This module handles loading and preprocessing of datasets with performance-based allocation.
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import BertTokenizer
from datasets import load_dataset, load_from_disk
import numpy as np
from collections import defaultdict

from config import ModelConfig, DatasetConfig, TrainingConfig

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

def load_trec_dataset():
    """Load TREC dataset
    
    Returns:
        Dataset: TREC dataset
    """
    dataset = load_from_disk("../trec")
    return dataset

def load_emotion_dataset():
    """Load Emotion dataset"""
    dataset = load_from_disk("../emotion")
    return dataset

def load_banking77_dataset():
    """Load Banking77 dataset"""
    dataset = load_from_disk("../banking77")
    return dataset

# Initialize tokenizer
pretrain_model_path = ModelConfig.PRETRAIN_MODEL_PATH
tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

def news_preprocess_function(examples):
    """Preprocess function for AG News
    
    Args:
        examples (dict): Batch of examples
        
    Returns:
        dict: Tokenized examples
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=DatasetConfig.NEWS_MAX_LENGTH)

def yahoo_preprocess_function(examples):
    """Preprocess function for Yahoo Answers
    
    Args:
        examples (dict): Batch of examples
        
    Returns:
        dict: Tokenized examples
    """
    texts = [f"{title} {content}" for title, content in 
             zip(examples["question_title"], examples["question_content"])]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=DatasetConfig.YAHOO_MAX_LENGTH)

def trec_preprocess_function(examples):
    """Preprocess function for TREC
    
    Args:
        examples (dict): Batch of examples
        
    Returns:
        dict: Tokenized examples
    """
    return tokenizer(examples['text'], 
                   padding='max_length',
                   truncation=True,
                   max_length=32)  # 超短序列

def emotion_preprocess_function(examples):
    return tokenizer(examples['text'], 
                     padding='max_length', 
                     truncation=True, 
                     max_length=DatasetConfig.EMOTION_MAX_LENGTH)

def banking77_preprocess_function(examples):
    return tokenizer(examples['text'], 
                     padding='max_length', 
                     truncation=True, 
                     max_length=DatasetConfig.BANKING77_MAX_LENGTH)

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
            "labels": torch.tensor(item[DatasetConfig.NEWS_LABLE_NAME]),
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
            "labels": torch.tensor(item[DatasetConfig.YAHOO_LABEL_NAME]),
        }

class TrecDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item["input_ids"]),
            'attention_mask': torch.tensor(item["attention_mask"]),
            # 'labels': torch.tensor(item[DatasetConfig.LABEL_NAME])  # 粗粒度6分类
            'labels': torch.tensor(item[DatasetConfig.LABEL_NAME])  # 细粒度50分类
        }

class EmotionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item["input_ids"]),
            'attention_mask': torch.tensor(item["attention_mask"]),
            'labels': torch.tensor(item[DatasetConfig.EMOTION_LABLE_NAME]),
        }

class Banking77Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item["input_ids"]),
            'attention_mask': torch.tensor(item["attention_mask"]),
            'labels': torch.tensor(item[DatasetConfig.BANKING77_LABLE_NAME]),
        }

def get_train_dataloaders(batch_size=8, client_id=0, performance_factor=1.0):
    """Get training DataLoader with performance-based allocation while maintaining label proportions
    
    Args:
        batch_size (int): Batch size
        client_id (int): Client identifier (for logging)
        performance_factor (float): Performance factor (0-1) determining data size
        
    Returns:
        DataLoader: Training data loader
    """
    dataset = load_emotion_dataset()
    train_data = dataset["train"]
    
    # 首先按标签分组数据
    label_indices = defaultdict(list)
    for idx, item in enumerate(train_data):
        label_indices[item[DatasetConfig.LABEL_NAME]].append(idx)
    
    # 计算每个客户端应分配的数据总量
    total_size = len(train_data)
    min_size = total_size * TrainingConfig.MIN_DATA_FACTOR  # Minimum of data
    max_size = total_size * TrainingConfig.MAX_DATA_FACTOR  # Maximum of data
    dataset_size = min(max_size, min_size + int((max_size - min_size) * performance_factor))
    
    # 按标签比例分配数据
    selected_indices = []
    for label, indices in label_indices.items():
        # 计算该标签在总数据集中的比例
        label_ratio = len(indices) / total_size
        # 计算该标签在客户端数据集中的数量
        label_count = max(1, int(dataset_size * label_ratio))
        # 随机选择该标签的数据
        selected_indices.extend(np.random.choice(indices, label_count, replace=False))
    
    print(f"📊 Client {client_id} (performance: {performance_factor:.1f}) allocated {len(selected_indices)} samples")
    
    # 创建子集
    train_subset = train_data.select(selected_indices).shuffle(seed=42)
    
    # 预处理和创建DataLoader
    train_data = train_subset.map(emotion_preprocess_function, batched=True)
    train_dataset = TrecDataset(train_data)
    
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def get_test_dataloaders(batch_size=16):
    """Get test DataLoader
    
    Args:
        batch_size (int): Batch size
        
    Returns:
        DataLoader: Test data loader
    """
    dataset = load_emotion_dataset()
    test_data = dataset["test"]
    test_data = test_data.map(emotion_preprocess_function, batched=True)
    test_dataset = TrecDataset(test_data)
    return DataLoader(test_dataset, batch_size=batch_size)