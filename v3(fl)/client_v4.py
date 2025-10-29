"""
Federated Learning Client with Synchronized Training
- Waits for server to be ready before starting
- Maintains sequential training order
- Handles connection synchronization
"""

import torch
import socket
import torch.nn as nn
from split_model import ModelPart0, ModelPart2
from dataset import get_train_dataloaders
from connect import *
from tqdm import tqdm
import time

class FLClient:
    def __init__(self, client_id=0, control_address=("localhost", 50010)):
        """
        Initialize federated learning client
        
        Args:
            client_id (int): Unique client identifier
            control_address (tuple): Server control address (host, port)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_id = client_id
        self.control_address = control_address
        self.train_port = None
        
        # Initialize model parts
        self.model_part0 = ModelPart0().to(self.device).eval()
        self.model_part2 = ModelPart2(num_classes=10).to(self.device).eval()
        self.criterion = nn.CrossEntropyLoss()
        
        # Performance factor (higher ID = better performance)
        self.performance_factor = (client_id + 1) / 10.0  # Range: 0.1 to 1.0

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

    def train(self, num_epochs=5):
        """
        Execute training loop
        - Handles connection synchronization
        - Manages full training cycle
        """
        if not self.register_with_server(3):  # Assuming 3 clients
            print(f"⚠️ Client {self.client_id} failed to register")
            return

        # Get data loader with performance-based allocation
        train_dataloader = get_train_dataloaders(
            batch_size=200,
            client_id=self.client_id,
            performance_factor=self.performance_factor
        )

        # Wait for training port to be ready
        waiting_server = False
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as train_socket:
                    train_socket.settimeout(30)
                    train_socket.connect(("localhost", self.train_port))
                    print(f"🚀 Client {self.client_id} starting training on port {self.train_port}")
                    
                    for epoch in range(num_epochs):
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
                              f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
                    
                    # Signal training completion
                    send_full_data(train_socket, {'training_complete': True})
                    print(f"🎉 Client {self.client_id} completed training")
                    break
                    
            except ConnectionRefusedError:
                if not waiting_server:
                    print(f"⏳ Client {self.client_id} waiting for server to complete evaluation...")
                    waiting_server = True
                time.sleep(5)
            except socket.timeout:
                print(f"⚠️ Training timeout for client {self.client_id}")
                break
            except Exception as e:
                print(f"⚠️ Training error for client {self.client_id}: {e}")
                break

if __name__ == "__main__":
    # Example: Sequential client execution
    clients = [FLClient(client_id=i) for i in range(3)]
    
    for client in clients:
        print(f"\n🚀 Starting client {client.client_id}")
        client.train(num_epochs=1)
        print(f"🛑 Client {client.client_id} finished")