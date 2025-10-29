import torch
import socket
import torch.nn as nn
import torch.optim as optim
from split_model import ModelPart0, ModelPart2
from connect import  *
from dataset import get_train_dataloaders  # 导入 TMDB 数据集

# todo: 多模型 + 聚合

# 连接服务器并训练
def client_process():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_part0 = ModelPart0().to(device)
    model_part2 = ModelPart2().to(device)
    optimizer = optim.AdamW(list(model_part0.parameters()) + list(model_part2.parameters()), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("localhost", 50010))

    # 获取数据加载器
    train_dataloader = get_train_dataloaders(batch_size=8)
    print("strat training")
    for epoch in range(10):
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Part0 计算
            hidden_states = model_part0(input_ids, attention_mask)

            # 发送到服务器
            send_full_data(client_socket, (hidden_states.cpu(), attention_mask.cpu()))
            #print("send hidden status0")

            # 接收服务器返回的 hidden_states1
            hidden_states1 = receive_full_data(client_socket).to(device)
            #print("receive hidden status1")

            # Part2 计算
            logits = model_part2(hidden_states1, attention_mask)
            loss = criterion(logits, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1} - Loss: {total_loss / len(train_dataloader):.4f}")

    client_socket .close()

if __name__ == "__main__":
    client_process()