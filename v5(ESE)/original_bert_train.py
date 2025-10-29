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
from dataset import get_train_dataloaders, get_test_dataloaders

# 完整BERT分类模型
class BertForClassification(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(ModelConfig.PRETRAIN_MODEL_PATH)
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
    # print("Classification Report:")
    # print(classification_report(all_labels, all_preds, target_names=[f"Class {i}" for i in range(DatasetConfig.DATASET_NUM_CLASSES)]))

    # 保存混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(SystemConfig.FULL_MODEL_SAVE_DIR, "confusion_matrix.png"))
    plt.close()
    
    return accuracy

# 训练函数
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SystemConfig.FULL_MODEL_SAVE_DIR, exist_ok=True)
    
    train_loader = get_train_dataloaders(batch_size = TrainingConfig.TRAIN_BATCH_SIZE)
    test_loader = get_test_dataloaders(batch_size = TrainingConfig.TEST_BATCH_SIZE)
    
    # 初始化模型
    model = BertForClassification(DatasetConfig.DATASET_NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    best_accuracy = 0
    train_loss_history = []
    train_acc_history = []
    
    for epoch in range(1, TrainingConfig.NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{TrainingConfig.NUM_EPOCHS}"):
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
        
        # 评估并保存模型
        test_acc = evaluate(model, test_loader, device)
        best_accuracy = test_acc
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(SystemConfig.FULL_MODEL_SAVE_DIR, f"fullbert_epoch-{epoch}_acc-{test_acc:.4f}_{timestamp}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Saved model with accuracy: {test_acc:.4f}")
    
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
    
    plt.savefig(os.path.join(SystemConfig.FULL_MODEL_SAVE_DIR, "training_curve.png"))
    plt.close()

if __name__ == "__main__":
    train()