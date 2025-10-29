import torch
import pickle
import socket
import struct
from split_model import ModelPart1, ModelPart0, ModelPart2
from connect import *
import os
from datetime import datetime
from dataset import get_test_dataloaders
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class Server:
    def __init__(self, preload_testset=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_part1 = ModelPart1().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model_part1.parameters(), lr=5e-5)
        self.best_accuracy = 0.0
        self.batch_size = 128
        self.batches = 0
        
        # 预加载测试集
        self.preload_testset = preload_testset
        self.test_dataloader = None
        if preload_testset:
            print("⏳ Preloading test dataset...")
            self.test_dataloader = list(get_test_dataloaders(batch_size=self.batch_size))  # 转换为list缓存
            print("✅ Test dataset preloaded")

    def save_model(self, accuracy):
        path = "./server_saved_models"
        os.makedirs(path, exist_ok=True)
        
        model_files = [f for f in os.listdir(path) if f.startswith("server_model_")]
        if len(model_files) >= 2:
            accuracies = [float(f.split("_")[3]) for f in model_files]
            min_acc_index = accuracies.index(min(accuracies))
            if accuracy > min(accuracies):
                os.remove(os.path.join(path, model_files[min_acc_index]))
            else:
                return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{path}/server_model_epoch_{self.epoch}_acc_{accuracy:.4f}_{timestamp}.pt"
        torch.save({
            'epoch': self.epoch,
            'accuracy': accuracy,
            'model_state_dict': self.model_part1.state_dict(),
        }, save_path)
        print(f"💾 Model saved to {save_path}")

    def run_test(self):
        print("\n🔍 Running evaluation...")
        model_part0 = ModelPart0().to(self.device).eval()
        model_part2 = ModelPart2(num_classes=10).to(self.device).eval()
        
        all_labels = []
        all_preds = []
        
        # 使用预加载的测试集或实时加载
        test_data = self.test_dataloader if self.preload_testset else get_test_dataloaders(batch_size=self.batch_size)
        
        with torch.no_grad():
            for batch in tqdm(test_data, desc="Testing"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                hidden_states0 = model_part0(input_ids, attention_mask)
                hidden_states1 = self.model_part1(hidden_states0, attention_mask)
                logits = model_part2(hidden_states1, attention_mask)
                
                _, preds = torch.max(logits, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        # 计算指标
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        accuracy = (all_labels == all_preds).mean() * 100
        
        # 动态类别处理
        num_classes = len(np.unique(all_labels))
        target_names = [f"Class {i}" for i in range(num_classes)]
        
        print("\n📊 Classification Report:")
        print(classification_report(all_labels, all_preds, 
                                  target_names=target_names,
                                  zero_division=0))
        
        # 可视化
        self.plot_results(all_labels, all_preds, num_classes)
        
        return accuracy

    def plot_results(self, labels, preds, num_classes):
        plt.figure(figsize=(max(10, num_classes//2), max(10, num_classes//2)))
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=range(num_classes),
                    yticklabels=range(num_classes))
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("server_confusion_matrix.png")
        plt.close()
        print("📈 Confusion matrix saved to server_confusion_matrix.png")

    def process(self, test_interval=5):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(("localhost", 50010))
        server_socket.listen(1)
        print("🌍 Server listening on port 50010...")
        
        try:
            while True:
                conn, addr = server_socket.accept()
                print(f"✅ Client {addr} connected!")
                
                try:
                    while True:
                        data = receive_full_data(conn)
                        if data is None:
                            print("Client disconnected")
                            break

                        # 训练步骤
                        hidden_states, attention_mask, labels = data
                        hidden_states = hidden_states.to(self.device)
                        attention_mask = attention_mask.to(self.device)
                        labels = labels.to(self.device)

                        self.model_part1.train()
                        self.optimizer.zero_grad()
                        hidden_states1 = self.model_part1(hidden_states, attention_mask)
                        send_full_data(conn, hidden_states1.detach().cpu())
                        
                        # 接收梯度
                        grad_data = receive_full_data(conn)
                        if grad_data is None:
                            print("No gradient received from client")
                            break
                        
                        hidden_states1.backward(gradient=grad_data.to(self.device))
                        self.optimizer.step()
                        self.batches += 1

                        # 定期测试
                        if self.batches / self.batch_size == test_interval:
                            self.batches = 0
                            test_accuracy = self.run_test()
                            if test_accuracy > self.best_accuracy:
                                self.best_accuracy = test_accuracy
                                self.save_model(test_accuracy)

                except (ConnectionResetError, BrokenPipeError) as e:
                    print(f"⚠️ Connection error: {e}")
                except Exception as e:
                    print(f"⚠️ Server error: {e}")
                finally:
                    conn.close()
                    print(f"❌ Client {addr} disconnected")

        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")
            final_accuracy = self.run_test()
            print(f"🏆 Final Test Accuracy: {final_accuracy:.2f}%")
        finally:
            server_socket.close()

if __name__ == "__main__":
    # 启动服务器，设置preload_testset=True来预加载测试集
    server = Server(preload_testset=True)
    server.process(test_interval=2)