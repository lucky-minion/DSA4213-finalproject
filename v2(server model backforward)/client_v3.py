import torch
import socket
import torch.nn as nn
from split_model import ModelPart0, ModelPart2
from connect import *
from dataset import get_train_dataloaders
from tqdm import tqdm

def client_process():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_part0 = ModelPart0().to(device).eval()  # 固定参数
    model_part2 = ModelPart2(num_classes=10).to(device).eval()  # 固定参数
    criterion = nn.CrossEntropyLoss()
    train_dataloader = get_train_dataloaders(batch_size=16)
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            client_socket.connect(("localhost", 50010))
            print("Connected to server - Starting training...")
            
            for epoch in range(20):  # 保留epoch循环
                total_loss = 0.0
                correct = 0
                total = 0
                
                # 使用tqdm显示进度条
                for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    # 前向传播
                    hidden_states = model_part0(input_ids, attention_mask)
                    send_full_data(client_socket, (hidden_states.cpu(), attention_mask.cpu(), labels.cpu()))

                    # 接收中间结果
                    hidden_states1 = receive_full_data(client_socket)
                    if hidden_states1 is None:
                        raise ConnectionError("Server disconnected")
                    hidden_states1 = hidden_states1.to(device).requires_grad_()

                    # 客户端计算
                    logits = model_part2(hidden_states1, attention_mask)
                    loss = criterion(logits, labels)
                    
                    # 计算准确率
                    _, predicted = torch.max(logits.data, 1)
                    batch_correct = (predicted == labels).sum().item()
                    correct += batch_correct
                    total += labels.size(0)
                    total_loss += loss.item()

                    # 每100个batch输出一次
                    if (batch_idx + 1) % 128 == 0:
                        batch_acc = 128 * batch_correct / labels.size(0)
                        avg_loss = total_loss / (batch_idx + 1)
                        epoch_acc = 128 * correct / total
                        print(f"\nEpoch {epoch+1} Batch {batch_idx+1}: "
                              f"Batch Loss: {loss.item():.4f} | "
                              f"Batch Acc: {batch_acc:.2f}% | "
                              f"Epoch Avg Loss: {avg_loss:.4f} | "
                              f"Epoch Acc: {epoch_acc:.2f}%")
                    
                    # 反向传播
                    loss.backward()
                    if hidden_states1.grad is not None:
                        send_full_data(client_socket, hidden_states1.grad.cpu())
                    else:
                        send_full_data(client_socket, None)

                # 每个epoch结束后输出总结
                epoch_loss = total_loss / len(train_dataloader)
                epoch_acc = 128 * correct / total
                print(f"\nEpoch {epoch+1} Summary: "
                      f"Avg Loss: {epoch_loss:.4f} | "
                      f"Accuracy: {epoch_acc:.2f}%")

        except socket.error as e:
            print(f"⚠️ Connection error: {e}")
        finally:
            print("Training completed.")

if __name__ == "__main__":
    client_process()