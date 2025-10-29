"""
Federated Learning Client with Synchronized Training
- Waits for server to be ready before starting
- Maintains sequential training order
- Handles connection synchronization
"""

import torch
import socket
import os
import torch.nn as nn
from datetime import datetime
import torch.optim as optim
from split_model import ModelPart0, ModelPart2
from dataset import get_train_dataloaders
from connect import *
from tqdm import tqdm
import time

from config import SystemConfig, ModelConfig, TrainingConfig, DatasetConfig

class FLClient:
    def __init__(self, client_id=0, control_address=("localhost", 50010)):
        """
        Initialize federated learning client
        
        Args:
            client_id (int): Unique client identifier
            control_address (tuple): Server control address (host, port)
        """
        self.device = SystemConfig.DEVICE
        self.client_id = client_id
        self.control_address = control_address
        self.train_port = None
        
        # Initialize model parts
        if SystemConfig.CLIENT_MODEL_PART_BACKFORWARD:
            self.model_part0 = ModelPart0().to(self.device).train()
            self.model_part2 = ModelPart2(num_classes=DatasetConfig.DATASET_NUM_CLASSES).to(self.device).train()
            self.optimizer = optim.AdamW(list(self.model_part0.parameters()) + list(self.model_part2.parameters()), lr=5e-5)
        else:
            self.model_part0 = ModelPart0().to(self.device).eval()
            self.model_part2 = ModelPart2(num_classes=DatasetConfig.DATASET_NUM_CLASSES).to(self.device).eval()

        self.criterion = nn.CrossEntropyLoss()
        self.curent_epoch = 0
        self.train_best_acc = 0
        # Performance factor (higher ID = better performance)
        self.performance_factor = (client_id + 1) / SystemConfig.CLIENT_NUM * 1.0  # Range: 0.1 to 1.0

    def register_with_server(self, total_clients):
        """
        Register with server and get training port assignment
        - Implements waiting logic if server is busy
        
        Args:
            total_clients (int): Total number of clients in system
            
        Returns:
            bool: True if registration successful
        """
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(10)
                    sock.connect(self.control_address)
                    
                    # Send registration request
                    send_full_data(sock, {'client_num': total_clients})
                    
                    # Get server response
                    response = receive_full_data(sock)
                    if response is None:
                        print("⚠️ No response from server")
                        return False
                    
                    if response.get('status') == 'ready':
                        self.train_port = self.control_address[1] + self.client_id + 1
                        print(f"Client {self.client_id} assigned port {self.train_port}")
                        return True
                    elif response.get('status') == 'error':
                        print(f"⚠️ Server error: {response.get('message')}")
                        return False
                        
            except ConnectionRefusedError:
                print(f"⏳ Client {self.client_id} waiting for server to be ready...")
                time.sleep(5)
            except socket.timeout:
                print(f"⚠️ Connection timeout for client {self.client_id}, retrying...")
                time.sleep(5)
            except Exception as e:
                print(f"⚠️ Registration error for client {self.client_id}: {e}")
                return False
    
    def save_model(self, accuracy):
        """
        Save model checkpoint with training metrics
        
        Args:
            server_id (int): ID of the server instance
            accuracy (float): Achieved test accuracy
        """
        path = "./client_saved_models"
        os.makedirs(path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        modelpart0_path = f"{path}/modelpart0/client {self.client_id}_epoch {self.curent_epoch}_acc {accuracy:.2f}_{timestamp}.pt"
        modelpart2_path = f"{path}/modelpart2/client {self.client_id}_epoch {self.curent_epoch}_acc {accuracy:.2f}_{timestamp}.pt"
        
        torch.save(self.model_part0.state_dict(), modelpart0_path)
        torch.save(self.model_part2.state_dict(), modelpart2_path)

    def train(self, num_epochs=5):
        """
        Execute training loop
        - Handles connection synchronization
        - Manages full training cycle
        """
        if not self.register_with_server(SystemConfig.CLIENT_NUM):  # client_num clients
            print(f"⚠️ Client {self.client_id} failed to register")
            return

        # Get data loader with performance-based allocation
        train_dataloader = get_train_dataloaders(
            batch_size=TrainingConfig.TRAIN_BATCH_SIZE,
            client_id=self.client_id,
            performance_factor=self.performance_factor
        )

        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as train_socket:
                    train_socket.settimeout(30)
                    train_socket.connect(("localhost", self.train_port))
                    print(f"🚀 Client {self.client_id} starting training on port {self.train_port}")
                    
                    for epoch in range(num_epochs):
                        self.curent_epoch += 1
                        total_loss = 0.0
                        correct = 0
                        total = 0
                        
                        for batch in tqdm(train_dataloader, 
                                         desc=f"Client {self.client_id} Epoch {epoch+1}"):
                            # Prepare batch data
                            input_ids = batch["input_ids"].to(self.device)
                            attention_mask = batch["attention_mask"].to(self.device)
                            labels = batch["labels"].to(self.device)
                            
                            # Client-side forward pass
                            hidden_states = self.model_part0(input_ids, attention_mask)
                            
                            # Send to server
                            send_full_data(train_socket, {
                                'client_id': self.client_id,
                                'hidden_states': hidden_states.cpu(),
                                'attention_mask': attention_mask.cpu(),
                                'labels': labels.cpu()
                            })
                            
                            # Get intermediate results
                            response = receive_full_data(train_socket)
                            if response is None:
                                raise ConnectionError("Server disconnected")
                            
                            hidden_states1 = response['hidden_states1'].to(self.device).requires_grad_()
                            
                            # Client-side computation
                            logits = self.model_part2(hidden_states1, attention_mask)
                            loss = self.criterion(logits, labels)
                            
                            # Backpropagation
                            loss.backward()
                            if SystemConfig.CLIENT_MODEL_PART_BACKFORWARD:
                                self.optimizer.step()
                            
                            # Send gradients back
                            grad_to_send = hidden_states1.grad.cpu() if hidden_states1.grad is not None else None
                            send_full_data(train_socket, {
                                'client_id': self.client_id,
                                'gradients': grad_to_send
                            })
                            
                            # Calculate metrics
                            _, predicted = torch.max(logits.data, 1)
                            correct += (predicted == labels).sum().item()
                            total += labels.size(0)
                            total_loss += loss.item()
                        
                        # Calculate and report epoch metrics
                        epoch_loss = total_loss / len(train_dataloader)
                        epoch_acc = 100 * correct / total
                        
                        send_full_data(train_socket, {
                            'epoch_metrics': {
                                'loss': epoch_loss,
                                'accuracy': epoch_acc,
                                'epoch': epoch + 1  # 1-based indexing
                            }
                        })
                        
                        print(f"Client {self.client_id} - Epoch {epoch+1}: "
                              f"Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
                        
                        if epoch_acc > self.train_best_acc:
                            self.save_model(epoch_acc)
                            self.train_best_acc = epoch_acc

                    # Signal training completion
                    send_full_data(train_socket, {'training_complete': True})
                    print(f"🎉 Client {self.client_id} completed training")
                    break
     
            except ConnectionRefusedError:
                time.sleep(5)
            except socket.timeout:
                print(f"⚠️ Training timeout for client {self.client_id}")
                break
            except Exception as e:
                print(f"⚠️ Training error for client {self.client_id}: {e}")
                break

if __name__ == "__main__":
    # Example: Sequential client execution
    clients = [FLClient(client_id=i) for i in range(SystemConfig.CLIENT_NUM)]
    
    for client in clients:
        print(f"\n🚀 Starting client {client.client_id}")
        client.train(num_epochs=TrainingConfig.NUM_EPOCHS)
        torch.cuda.empty_cache()
        print(f"🛑 Client {client.client_id} finished")