import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from datasets import load_dataset

# 1. 加载 IMDb 数据集
def load_tmdb_dataset():
    dataset = load_dataset("imdb")  # TMDB 电影评论数据集可以替换为 "tmdb"
    return dataset

# 2. 定义 Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# 3. 预处理函数
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# 4. 转换为 PyTorch Dataset
class IMDBDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item["label"]),  # 修正这里，IMDb 数据的类别是 "label"
        }

# 5. 获取数据加载器
def get_train_dataloaders(batch_size=8):
    dataset = load_tmdb_dataset()

    train_data = dataset["train"].shuffle(seed=42)
    train_data = train_data.map(preprocess_function, batched=True)

    train_dataset = IMDBDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader