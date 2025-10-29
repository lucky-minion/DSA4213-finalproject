import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

from config import ModelConfig, DatasetConfig, TrainingConfig, SystemConfig

# 配置参数
class Config:
    pretrain_model_path = ModelConfig.PRETRAIN_MODEL_PATH
    dataset_path = DatasetConfig.DATASET_PATH
    num_classes = DatasetConfig.DATASET_NUM_CLASSES
    batch_size = TrainingConfig.TRAIN_BATCH_SIZE
    num_epochs = TrainingConfig.NUM_EPOCHS
    learning_rate = TrainingConfig.LEARNING_RATE
    save_dir = SystemConfig.FULL_MODEL_SAVE_DIR

dataset = load_from_disk(Config.dataset_path)
tokenizer = BertTokenizer.from_pretrained(Config.pretrain_model_path)

# 完整BERT分类模型
class BertForClassification(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(Config.pretrain_model_path)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

# 数据集处理
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
            "labels": torch.tensor(item["topic"])
        }

class TrecDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids']),
            'attention_mask': torch.tensor(item['attention_mask']),
            'labels': torch.tensor(item['coarse_label'])  # 粗粒度6分类
        }

# 预处理函数
def yahoo_preprocess_function(examples):
    tokenizer = BertTokenizer.from_pretrained(Config.pretrain_model_path)
    texts = [f"{title} {content}" for title, content in 
             zip(examples["question_title"], examples["question_content"])]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

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

# 评估函数
def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask)
            _, preds = torch.max(logits, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    accuracy = (np.array(all_labels) == np.array(all_preds)).mean()
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=[f"Class {i}" for i in range(Config.num_classes)]))

    # 保存混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(Config.save_dir, "confusion_matrix.png"))
    plt.close()
    
    return accuracy

# 训练函数
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(Config.save_dir, exist_ok=True)
    
    # 加载数据
    train_data = dataset["train"].map(trec_preprocess_function, batched=True)
    test_data = dataset["test"].map(trec_preprocess_function, batched=True)
    
    train_dataset = TrecDataset(train_data)
    test_dataset = TrecDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size)
    
    # 初始化模型
    model = BertForClassification(Config.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    best_accuracy = 0
    train_loss_history = []
    train_acc_history = []
    
    for epoch in range(Config.num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.num_epochs}"):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)
        
        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        
        # 评估并保存最佳模型
        test_acc = evaluate(model, test_loader, device)
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(Config.save_dir, f"best_model_acc{test_acc:.4f}_{timestamp}.pt")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'accuracy': test_acc,
                'loss_history': train_loss_history,
                'acc_history': train_acc_history
            }, model_path)
            print(f"Saved best model with accuracy: {test_acc:.4f}")
    
    # 保存训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.savefig(os.path.join(Config.save_dir, "training_curve.png"))
    plt.close()

if __name__ == "__main__":
    train()