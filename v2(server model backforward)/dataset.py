import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from datasets import load_dataset, load_from_disk

# 修改为加载多分类数据集，例如 AG News
def load_ag_news_dataset():
    dataset = load_from_disk("../ag_news_dataset")  # AG News 是一个4类新闻分类数据集
    return dataset

def load_yahoo_dataset():
    dataset = load_from_disk("../yahoo_answers_topics")  # 28个类别
    return dataset

# 2. 定义 Tokenizer
pretrain_model_path = '../bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

# 3. 预处理函数
def news_preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

def yahoo_preprocess_function(examples):
     # 将问题和内容组合起来
    texts = [f"{title} {content}" for title, content in zip(examples["question_title"], examples["question_content"])]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

# 4. 转换为 PyTorch Dataset
class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item["label"]),  # AG News 的类别是 "label"
        }

class YahooDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item["topic"]),  # 使用topic作为标签
        }

# 5. 获取数据加载器
def get_train_dataloaders(batch_size=8):
    #dataset = load_ag_news_dataset()
    dataset = load_yahoo_dataset()
    train_data = dataset["train"]
    

    # 获取数据集总大小
    dataset_size = len(train_data) // 50
    train_data = train_data.select(range(dataset_size)).shuffle(seed=42)

    #train_data = dataset["train"].shuffle(seed=42)

    #train_data = train_data.map(news_preprocess_function, batched=True)
    train_data = train_data.map(yahoo_preprocess_function, batched=True)

    #train_dataset = NewsDataset(train_data)
    train_dataset = YahooDataset(train_data)

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def get_test_dataloaders(batch_size=8):
    dataset = load_yahoo_dataset()
    test_data = dataset["test"]
    test_data = test_data.map(yahoo_preprocess_function, batched=True)
    test_dataset = YahooDataset(test_data)
    return DataLoader(test_dataset, batch_size=batch_size)