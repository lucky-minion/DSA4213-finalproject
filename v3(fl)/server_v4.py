"""
Federated Learning Server with Synchronized Evaluation
- Manages client connections sequentially
- Ensures evaluation completes before next client starts
- Maintains training visualization and model saving
"""

import torch
import socket
import os
import threading
from datetime import datetime
from collections import defaultdict
from split_model import ModelPart1, ModelPart0, ModelPart2
from dataset import get_test_dataloaders
from connect import *
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import queue

class FLServer:
    def __init__(self, preload_testset=True, control_port=50010):
        """
        Initialize the federated learning server
        
        Args:
            preload_testset (bool): Whether to load test data at startup
            control_port (int): Base port for server communications
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.control_port = control_port
        self.client_num = 0
        self.server_instances = {}
        self.train_queue = queue.Queue()
        self.lock = threading.Lock()
        
        # Training visualization storage
        self.train_loss_history = defaultdict(list)
        self.train_acc_history = defaultdict(list)
        self.test_acc_history = defaultdict(list)
        
        # Evaluation synchronization
        self.evaluation_complete = threading.Event()
        self.evaluation_complete.set()  # Initially set to True
        self.currently_training = False
        
        # Test data handling
        self.preload_testset = preload_testset
        self.test_dataloader = None
        if preload_testset:
            self._preload_test_data()

    def _preload_test_data(self):
        """Load test dataset into memory for faster evaluation"""
        print("⏳ Preloading test dataset...")
        self.test_dataloader = list(get_test_dataloaders(batch_size=200))
        print("✅ Test dataset preloaded")

    def save_model(self, server_id, accuracy):
        """
        Save model checkpoint with training metrics
        
        Args:
            server_id (int): ID of the server instance
            accuracy (float): Achieved test accuracy
        """
        path = "./server_saved_models"
        os.makedirs(path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{path}/server_{server_id}_epoch{self.server_instances[server_id]['current_epoch']}_acc{accuracy:.2f}_{timestamp}.pt"
        
        torch.save({
            'epoch': self.server_instances[server_id]['current_epoch'],
            'accuracy': accuracy,
            'model_state_dict': self.server_instances[server_id]['model'].state_dict(),
            'train_loss': self.train_loss_history[server_id],
            'train_acc': self.train_acc_history[server_id],
            'test_acc': self.test_acc_history[server_id]
        }, model_path)
        
        print(f"💾 Saved model for server {server_id} to {model_path}")
        self._save_training_plots(server_id)

    def _save_training_plots(self, server_id):
        """Generate and save training visualization plots"""
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_history[server_id], label='Training Loss')
        plt.title(f'Server {server_id} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc_history[server_id], label='Training Accuracy')
        plt.plot(self.test_acc_history[server_id], label='Test Accuracy')
        plt.title(f'Server {server_id} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plot_path = f"./server_saved_models/server_{server_id}_training_plot.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"📊 Saved training plots for server {server_id}")

    def evaluate_model(self, server_id):
        """
        Evaluate model on test set and save metrics
        
        Args:
            server_id (int): ID of the server instance to evaluate
            
        Returns:
            float: Test accuracy
        """
        self.evaluation_complete.clear()  # Signal evaluation start
        
        try:
            print(f"\n🔍 Evaluating server {server_id} model...")
            model_part0 = ModelPart0().to(self.device).eval()
            model_part2 = ModelPart2(num_classes=10).to(self.device).eval()
            server_model = self.server_instances[server_id]['model'].eval()
            
            all_labels = []
            all_preds = []
            
            with torch.no_grad():
                for batch in tqdm(self.test_dataloader, desc=f"Evaluating"):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    hidden_states0 = model_part0(input_ids, attention_mask)
                    hidden_states1 = server_model(hidden_states0, attention_mask)
                    logits = model_part2(hidden_states1, attention_mask)
                    
                    _, preds = torch.max(logits, 1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
            
            # Calculate and store metrics
            accuracy = (np.array(all_labels) == np.array(all_preds)).mean() * 100
            self.test_acc_history[server_id].append(accuracy)
            
            # Save confusion matrix
            self._save_confusion_matrix(all_labels, all_preds, server_id)
            
            # Print classification report
            print(f"\n📊 Classification Report for Server {server_id}:")
            print(classification_report(all_labels, all_preds, 
                                      target_names=[f"Class {i}" for i in range(10)],
                                      zero_division=0))
            
            return accuracy
            
        finally:
            self.evaluation_complete.set()  # Signal evaluation complete
            print(f"✅ Evaluation complete for server {server_id}")

    def _save_confusion_matrix(self, labels, preds, server_id):
        """Generate and save confusion matrix visualization"""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Server {server_id} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        
        cm_path = f"./server_saved_models/server_{server_id}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"📈 Saved confusion matrix for server {server_id}")

    def handle_control_connection(self, conn, addr):
        """
        Handle initial client control connection
        - Registers clients
        - Assigns training ports
        - Ensures proper synchronization
        """
        try:
            print(f"🎛️ Control connection from {addr}")
            data = receive_full_data(conn)
            if data is None or 'client_num' not in data:
                raise ValueError("Invalid client_num received")
            
            # Wait for any ongoing evaluation to complete
            if not self.evaluation_complete.wait(timeout=60):
                raise TimeoutError("Evaluation timeout")
            
            with self.lock:
                self.client_num = data['client_num']
                
                # Initialize server instances if needed
                for i in range(self.client_num):
                    server_id = i + 1
                    if server_id not in self.server_instances:
                        self.server_instances[server_id] = {
                            'model': ModelPart1().to(self.device),
                            'optimizer': torch.optim.AdamW(ModelPart1().parameters(), lr=5e-5),
                            'status': 'idle',
                            'best_accuracy': 0.0,
                            'current_epoch': 0
                        }
                        self.train_queue.put(server_id)
                        # Initialize visualization data
                        self.train_loss_history[server_id] = []
                        self.train_acc_history[server_id] = []
                        self.test_acc_history[server_id] = []
                
                # Send port assignments to client
                send_full_data(conn, {
                    'status': 'ready',
                    'server_ports': [self.control_port + i + 1 for i in range(self.client_num)]
                })
                
        except Exception as e:
            print(f"⚠️ Control connection error: {e}")
            send_full_data(conn, {'status': 'error', 'message': str(e)})
        finally:
            conn.close()

    def train_client(self, server_id):
        """
        Handle training for a single client
        - Manages the full training cycle
        - Ensures proper synchronization
        """
        train_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        train_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        train_port = self.control_port + server_id
        train_socket.bind(("localhost", train_port))
        train_socket.listen(1)
        
        try:
            print(f"🔄 Starting training server {server_id} on port {train_port}")
            conn, addr = train_socket.accept()
            print(f"✅ Client connected to server {server_id}")

            server_instance = self.server_instances[server_id]
            server_instance['status'] = 'training'
            self.currently_training = True

            while True:
                data = receive_full_data(conn)
                if data is None:
                    break

                if 'epoch_metrics' in data:
                    # Store training metrics
                    epoch_num = data['epoch_metrics']['epoch']
                    loss = data['epoch_metrics']['loss']
                    accuracy = data['epoch_metrics']['accuracy']
                    
                    server_instance['current_epoch'] = epoch_num
                    self.train_loss_history[server_id].append(loss)
                    self.train_acc_history[server_id].append(accuracy)
                    
                    print(f"📊 Server {server_id} - Epoch {epoch_num}: "
                          f"Loss: {loss:.4f} | Acc: {accuracy:.2f}%")

                elif 'training_complete' in data:
                    print(f"🏁 Training complete for server {server_id}")
                    
                    # Perform final evaluation
                    test_accuracy = self.evaluate_model(server_id)
                    
                    # Save model if improved
                    if test_accuracy > server_instance['best_accuracy']:
                        server_instance['best_accuracy'] = test_accuracy
                        self.save_model(server_id, test_accuracy)
                    
                    break

                elif 'hidden_states' in data:
                    # Standard training step
                    server_instance['model'].train()
                    server_instance['optimizer'].zero_grad()
                    
                    hidden_states = data['hidden_states'].to(self.device)
                    attention_mask = data['attention_mask'].to(self.device)
                    labels = data['labels'].to(self.device)
                    
                    # Forward pass
                    hidden_states1 = server_instance['model'](hidden_states, attention_mask)
                    send_full_data(conn, {'hidden_states1': hidden_states1.detach().cpu()})
                    
                    # Backward pass
                    grad_data = receive_full_data(conn)
                    if 'gradients' in grad_data:
                        gradients = grad_data['gradients'].to(self.device)
                        hidden_states1.backward(gradient=gradients)
                        server_instance['optimizer'].step()

        except Exception as e:
            print(f"⚠️ Training error for server {server_id}: {e}")
        finally:
            server_instance['status'] = 'idle'
            self.currently_training = False
            conn.close()
            train_socket.close()
            print(f"🛑 Training server {server_id} closed")

    def control_thread(self):
        """Thread to handle control plane connections"""
        control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        control_socket.bind(("localhost", self.control_port))
        control_socket.listen(5)
        print(f"🌍 Control server started on port {self.control_port}")

        try:
            while True:
                conn, addr = control_socket.accept()
                threading.Thread(target=self.handle_control_connection, args=(conn, addr)).start()
        except KeyboardInterrupt:
            print("🛑 Control thread shutting down...")
        finally:
            control_socket.close()

    def training_thread(self):
        """Thread to manage client training sequence"""
        while True:
            server_id = self.train_queue.get()
            self.train_client(server_id)
            self.train_queue.task_done()

    def run(self):
        """Main server execution loop"""
        # Start control and training threads
        control_thread = threading.Thread(target=self.control_thread, daemon=True)
        training_thread = threading.Thread(target=self.training_thread, daemon=True)
        
        control_thread.start()
        training_thread.start()
        
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")
            control_thread.join()
            training_thread.join()

if __name__ == "__main__":
    server = FLServer(preload_testset=True)
    server.run()