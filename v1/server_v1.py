import flwr as fl
import torch
import pickle
import socket
import struct
from split_model import ModelPart1
from connect import  *


def server_process():
    """服务器端，处理 Part1"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_part1 = ModelPart1().to(device)
    model_part1.eval()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("localhost", 50010))
    server_socket.listen(1)
    print("🌍 Server listening on port 50010...")
    conn, addr = server_socket.accept()
    print("✅ Client", addr, "connected!")

    while True:
        try:
            # 1️ 接收数据
            hidden_states, attention_mask = receive_full_data(conn)
            #print("reveive hidden status0")

            # 2️ 服务器端计算
            hidden_states1 = model_part1(hidden_states.to(device), attention_mask.to(device))

            # 3️ 发送数据回客户端
            send_full_data(conn, hidden_states1.cpu())
            #print("send hidden status1")

        except Exception as e:
            print(f"⚠️ 服务器错误: {e}")
            break

    conn.close()
    server_socket.close()

if __name__ == "__main__":
    server_process()
