import torch
import os
import sys
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from dataset import get_test_dataloaders
from split_model import ModelPart0, ModelPart1, ModelPart2
from config import SystemConfig, ModelConfig, DatasetConfig, TrainingConfig

def load_combined_model(model_part0_path, model_part1_path, model_part2_path, device):
    """加载分割模型的三个部分并组合成完整模型"""
    # 初始化各部分模型
    model_part0 = ModelPart0().to(device)
    model_part1 = ModelPart1().to(device)
    model_part2 = ModelPart2(num_classes=DatasetConfig.DATASET_NUM_CLASSES).to(device)
    
    # 加载预训练权重
    if model_part1_path != None:
        model_part0.load_state_dict(torch.load(model_part0_path, map_location=device))
    if model_part1_path != None:
        model_part1.load_state_dict(torch.load(model_part1_path, map_location=device))
    model_part2.load_state_dict(torch.load(model_part2_path, map_location=device))
    
    # 设置为评估模式
    model_part0.eval()
    model_part1.eval()
    model_part2.eval()
    
    return model_part0, model_part1, model_part2

def evaluate_model(model_part0, model_part1, model_part2, test_loader, device, save_dir=None):
    """评估模型性能并生成可视化结果"""
    all_labels = []
    all_preds = []

    # 检查聚合后的模型输出
    sample = next(iter(test_loader))
    input_ids = sample["input_ids"].to(device)
    attention_mask = sample["attention_mask"].to(device)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 前向传播（分三部分）
            hidden_states0 = model_part0(input_ids, attention_mask)
            hidden_states1 = model_part1(hidden_states0, attention_mask)
            logits = model_part2(hidden_states1, attention_mask)
            
            _, preds = torch.max(logits, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # 计算准确率
    accuracy = (np.array(all_labels) == np.array(all_preds)).mean()
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # 分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                               target_names=[f"Class {i}" for i in range(DatasetConfig.DATASET_NUM_CLASSES)], zero_division=0))
    
    # 保存结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存混淆矩阵
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(all_labels, all_preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=[f"Class {i}" for i in range(DatasetConfig.DATASET_NUM_CLASSES)],
                    yticklabels=[f"Class {i}" for i in range(DatasetConfig.DATASET_NUM_CLASSES)])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        cm_path = os.path.join(save_dir, f"confusion_matrix_{timestamp}.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"Saved confusion matrix to {cm_path}")
        
        # 保存分类报告为文本文件
        report = classification_report(all_labels, all_preds, 
                                     target_names=[f"Class {i}" for i in range(DatasetConfig.DATASET_NUM_CLASSES)], zero_division=0)
        report_path = os.path.join(save_dir, f"classification_report_{timestamp}.txt")
        with open(report_path, 'w') as f:
            f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
            f.write(report)
        print(f"Saved classification report to {report_path}")
    
    return accuracy

def main():
    # 配置参数
    device = SystemConfig.DEVICE
    save_dir = "./evaluation_results"  # 结果保存目录
    
    # 模型路径 - 请根据实际情况修改
    model_part0_path = "./aggregated_models/weighted_part0.pt"
    model_part1_path = None
    model_part2_path = "./aggregated_models/weighted_part2.pt"
    
    # 加载测试数据
    test_loader = get_test_dataloaders(batch_size=TrainingConfig.TEST_BATCH_SIZE)
    
    # 加载模型
    print("Loading trained models...")
    model_part0, model_part1, model_part2 = load_combined_model(
        model_part0_path, model_part1_path, model_part2_path, device
    )
    
    # 评估模型
    print("\nEvaluating model on test set...")
    accuracy = evaluate_model(model_part0, model_part1, model_part2, test_loader, device, save_dir)

if __name__ == "__main__":
    main()