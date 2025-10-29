import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from dataset import get_test_dataloaders
from split_model import ModelPart0, ModelPart1, ModelPart2
from tqdm import tqdm
import os

def load_server_model(model_path):
    """加载服务器端保存的最优模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_part1 = ModelPart1().to(device)
    
    # 如果传入的是目录，自动选择最优模型
    if os.path.isdir(model_path):
        model_files = [f for f in os.listdir(model_path) if f.startswith("server_model_")]
        if not model_files:
            raise ValueError("No server models found in the directory")
        # 按准确率排序
        model_files.sort(key=lambda x: float(x.split("_")[3]), reverse=True)
        model_path = os.path.join(model_path, model_files[0])
    
    checkpoint = torch.load(model_path)
    model_part1.load_state_dict(checkpoint['model_state_dict'])
    model_part1.eval()
    return model_part1

def evaluate_local(model_path="./server_saved_models"):
    """本地评估函数，不依赖客户端-服务器通信"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化所有模型部分
    model_part0 = ModelPart0().to(device).eval()
    model_part1 = load_server_model(model_path).to(device).eval()
    model_part2 = ModelPart2(num_classes=28).to(device).eval()
    
    # 获取测试数据
    test_dataloader = get_test_dataloaders(batch_size=32)
    
    # 准备存储预测结果
    all_labels = []
    all_preds = []
    
    print("Starting local evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 完整的前向传播
            hidden_states0 = model_part0(input_ids, attention_mask)
            hidden_states1 = model_part1(hidden_states0, attention_mask)
            logits = model_part2(hidden_states1, attention_mask)
            
            _, preds = torch.max(logits, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # 转换为numpy数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # 计算并打印分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=[f"Class {i}" for i in range(28)]))
    
    # 绘制混淆矩阵
    plt.figure(figsize=(20, 20))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=[f"{i}" for i in range(28)],
                yticklabels=[f"{i}" for i in range(28)])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix.png")
    plt.show()
    
    # 绘制准确率柱状图
    class_acc = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(15, 5))
    plt.bar(range(28), class_acc)
    plt.title("Class-wise Accuracy")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.xticks(range(28))
    plt.grid(True, axis='y')
    plt.savefig("class_accuracy.png")
    plt.show()

if __name__ == "__main__":
    # 可以传入模型文件路径或包含模型的目录
    evaluate_local(model_path="./server_saved_models")