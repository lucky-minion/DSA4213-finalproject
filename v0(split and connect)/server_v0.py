import flwr as fl
import torch
import pickle
import socket
import torch.nn as nn
from transformers import BertModel

# 服务器端的 Part1 模型
class ModelPart1(nn.Module):
    def __init__(self):
        super(ModelPart1, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.encoder_layers = self.bert.encoder.layer[1:-1]  # 第 1 层到倒数第 2 层

    def forward(self, hidden_states, attention_mask):
        attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float)
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        return hidden_states

# 启动服务器，监听客户端
def start_server():
    model_part1 = ModelPart1()
    model_part1.eval()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("localhost", 50010))
    server_socket.listen(5)
    print("Server listening on port 50010...")

    conn, _ = server_socket.accept()
    print("Client connected!")

    while True:
        data = conn.recv(24096)
        if not data:
            break
        hidden_states, attention_mask = pickle.loads(data)  # 解码数据

        with torch.no_grad():
            hidden_states1 = model_part1(hidden_states, attention_mask)

        conn.sendall(pickle.dumps(hidden_states1))  # 发送处理后的隐藏状态

    conn.close()
    server_socket.close()

if __name__ == "__main__":
    start_server()
