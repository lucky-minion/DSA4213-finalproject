import flwr as fl
import torch
import pickle
import socket
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer

# 客户端的 Part0 和 Part2
class ModelPart0(nn.Module):
    def __init__(self):
        super(ModelPart0, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.embeddings = self.bert.embeddings
        self.encoder_layer_0 = self.bert.encoder.layer[0]

    def forward(self, input_ids, attention_mask):
        x = self.embeddings(input_ids)
        attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float)
        x = self.encoder_layer_0(x, attention_mask=attention_mask)[0]
        return x

class ModelPart2(nn.Module):
    def __init__(self):
        super(ModelPart2, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.encoder_last = self.bert.encoder.layer[-1]
        self.pooler = self.bert.pooler
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, hidden_states, attention_mask):
        attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float)
        hidden_states = self.encoder_last(hidden_states, attention_mask=attention_mask)[0]
        pooled_output = self.pooler(hidden_states)
        return self.classifier(pooled_output)

# 连接服务器并训练
def train_client():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_part0 = ModelPart0().to(device)
    model_part2 = ModelPart2().to(device)
    optimizer = optim.AdamW(list(model_part0.parameters()) + list(model_part2.parameters()), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    sentences = ["This movie was great!", "I hated this film."]
    labels = torch.tensor([1, 0]).to(device)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.connect(("localhost", 50010))

    for epoch in range(20):
        total_loss = 0
        for i, sentence in enumerate(sentences):
            optimizer.zero_grad()

            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # 1. Part0 计算
            hidden_states = model_part0(input_ids, attention_mask)

            # 2. 发送 hidden_states 到服务器
            server_socket.sendall(pickle.dumps((hidden_states.cpu(), attention_mask.cpu())))
            
            # 3. 接收服务器返回的 hidden_states1
            data = server_socket.recv(4000)
            hidden_states1 = pickle.loads(data).to(device)

            # 4. Part2 计算
            logits = model_part2(hidden_states1, attention_mask)
            loss = criterion(logits, labels[i].unsqueeze(0))

            # 5. 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1} - Loss: {total_loss / len(sentences):.4f}")

    server_socket.close()

if __name__ == "__main__":
    train_client()
