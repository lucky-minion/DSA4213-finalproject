import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from dataset import get_test_dataloaders
from split_model import ModelPart0, ModelPart1, ModelPart2
from original_bert_train import BertForClassification, Config
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_full_model(model_path, num_classes):
    """加载完整的BERT模型"""
    model = BertForClassification(num_classes)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def evaluate_full_model(model, dataloader, device):
    """评估完整BERT模型"""
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask)
            _, preds = torch.max(logits, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    return all_labels, all_preds

def evaluate_federated_model(part1_model, dataloader, device):
    """评估联邦学习模型（使用split_model中的Part0和Part2）"""
    # 直接使用split_model中的类
    part0 = ModelPart0().to(device).eval()
    part2 = ModelPart2(Config.num_classes).to(device).eval()
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 联邦模型前向传播
            hidden0 = part0(input_ids, attention_mask)
            hidden1 = part1_model(hidden0, attention_mask)
            logits = part2(hidden1, attention_mask)
            
            _, preds = torch.max(logits, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    return all_labels, all_preds

def save_confusion_matrix(labels, preds, model_type):
    """保存混淆矩阵"""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix ({model_type})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    os.makedirs("./comparison_results", exist_ok=True)
    plt.savefig(f"./comparison_results/confusion_matrix_{model_type}.png")
    plt.close()

def compare_models(full_model_path, part1_model_path):
    """比较完整模型和联邦模型性能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载测试数据
    test_loader = list(get_test_dataloaders(batch_size=32))
    
    # 1. 评估完整BERT模型
    print("Loading and evaluating Full BERT Model...")
    full_model = load_full_model(full_model_path, Config.num_classes).to(device)
    full_labels, full_preds = evaluate_full_model(full_model, test_loader, device)
    full_acc = (np.array(full_labels) == np.array(full_preds)).mean()
    
    # 保存完整模型的混淆矩阵
    save_confusion_matrix(full_labels, full_preds, "full_model")
    
    # 2. 评估联邦BERT模型
    print("\nLoading and evaluating Federated BERT Model...")
    part1_model = ModelPart1().to(device)
    part1_model.load_state_dict(torch.load(part1_model_path))
    fed_labels, fed_preds = evaluate_federated_model(part1_model, test_loader, device)
    fed_acc = (np.array(fed_labels) == np.array(fed_preds)).mean()
    
    # 保存联邦模型的混淆矩阵
    save_confusion_matrix(fed_labels, fed_preds, "federated_model")
    
    # 3. 打印对比结果
    print("\n=== Model Comparison Results ===")
    print(f"Full BERT Model Accuracy:    {full_acc:.4f}")
    print(f"Federated BERT Model Accuracy: {fed_acc:.4f}")
    print(f"Accuracy Difference:         {abs(full_acc-fed_acc):.4f}")
    
    # 4. 打印分类报告
    print("\n=== Full Model Classification Report ===")
    print(classification_report(full_labels, full_preds, 
                              target_names=[f"Class {i}" for i in range(10)],
                              digits=4))
    
    print("\n=== Federated Model Classification Report ===")
    print(classification_report(fed_labels, fed_preds,
                              target_names=[f"Class {i}" for i in range(10)],
                              digits=4))
    
    # 5. 绘制准确率对比图
    plt.figure(figsize=(8, 5))
    bars = plt.bar(['Full BERT', 'Federated BERT'], [full_acc, fed_acc], color=['blue', 'orange'])
    plt.title('Model Accuracy Comparison', pad=20)
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # 在柱子上显示准确率数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    os.makedirs("./comparison_results", exist_ok=True)
    plt.savefig('./comparison_results/accuracy_comparison.png', bbox_inches='tight')
    plt.close()
    
    print("\nComparison results saved to ./comparison_results/ directory")

if __name__ == "__main__":
    # 需要替换为实际的模型路径
    full_model_path = "./full_bert_models/best_model_acc0.9231_20230615.pt"
    part1_model_path = "./server_saved_models/server_part1_best.pt"
    
    compare_models(full_model_path, part1_model_path)