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
from split_model import ModelPart0, ModelPart1, ModelPart2
from dataset import get_train_dataloaders
from connect import *
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

from config import SystemConfig, ModelConfig, TrainingConfig, DatasetConfig

class FLClient:
    def __init__(self, client_id=0):
        """
        Initialize federated learning client
        
        Args:
            client_id (int): Unique client identifier
            control_address (tuple): Server control address (host, port)
        """
        self.device = SystemConfig.DEVICE
        self.client_id = client_id

        self.model_part1 = ModelPart1().to(self.device).eval()

        # Initialize model parts
        if SystemConfig.CLIENT_MODEL_PART_BACKFORWARD:
            self.model_part0 = ModelPart0().to(self.device).eval()
            self.model_part2 = ModelPart2(num_classes=DatasetConfig.DATASET_NUM_CLASSES).to(self.device).train()
            self.optimizer = optim.AdamW(self.model_part2.parameters(), lr=TrainingConfig.LEARNING_RATE)
        else:
            self.model_part0 = ModelPart0().to(self.device).eval()
            self.model_part2 = ModelPart2(num_classes=DatasetConfig.DATASET_NUM_CLASSES).to(self.device).eval()

        self.criterion = nn.CrossEntropyLoss()
        self.curent_epoch = 0
        self.train_best_acc = 0
        # Performance factor (higher ID = better performance)
        self.performance_factor = (client_id + 1) / SystemConfig.CLIENT_NUM * 1.0  # Range: 0.1 to 1.0
        
        # For plotting
        self.train_loss_history = []
        self.train_acc_history = []

    def save_model(self, accuracy):
        """
        Save model checkpoint with training metrics
        
        Args:
            server_id (int): ID of the server instance
            accuracy (float): Achieved test accuracy
        """
        path = "./client_saved_models"
        os.makedirs(path, exist_ok=True)
        os.makedirs(f"{path}/modelpart2", exist_ok=True)
        os.makedirs(f"{path}/plots", exist_ok=True)  # Create directory for plots
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        modelpart2_path = f"{path}/modelpart2/client-{self.client_id}_epoch-{self.curent_epoch}_acc-{accuracy:.2f}_{timestamp}.pt"
        
        torch.save(self.model_part2.state_dict(), modelpart2_path)

    def save_training_plots(self):
        """Save training loss and accuracy plots"""
        path = "./client_saved_models/plots"
        os.makedirs(path, exist_ok=True)
        
        # Plot and save training loss
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss_history, label='Training Loss')
        plt.title(f'Client {self.client_id} - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{path}/client_{self.client_id}_loss.png")
        plt.close()
        
        # Plot and save training accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_acc_history, label='Training Accuracy')
        plt.title(f'Client {self.client_id} - Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.savefig(f"{path}/client_{self.client_id}_accuracy.png")
        plt.close()

    def train(self, num_epochs=5):
        """
        Execute training loop
        - Handles connection synchronization
        - Manages full training cycle
        """

        # Get data loader with performance-based allocation
        train_dataloader = get_train_dataloaders(
            batch_size=TrainingConfig.TRAIN_BATCH_SIZE,
            client_id=self.client_id,
            performance_factor=self.performance_factor
        )

        for epoch in range(num_epochs):
            self.curent_epoch += 1
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch in tqdm(train_dataloader,
                                desc=f"Client {self.client_id} Epoch {epoch+1}"):
                # Prepare batch data
                self.optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Client-side forward pass
                hidden_states = self.model_part0(input_ids, attention_mask)
                hidden_states1 = self.model_part1(hidden_states, attention_mask)
                
                # Client-side computation
                logits = self.model_part2(hidden_states1, attention_mask)
                loss = self.criterion(logits, labels)
                        
                # Backpropagation
                loss.backward()
                if SystemConfig.CLIENT_MODEL_PART_BACKFORWARD:
                    self.optimizer.step()
                
                # Calculate metrics
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
                        
            # Calculate and report epoch metrics
            epoch_loss = total_loss / len(train_dataloader)
            epoch_acc = 100 * correct / total
            
            # Store metrics for plotting
            self.train_loss_history.append(epoch_loss)
            self.train_acc_history.append(epoch_acc)
                        
            print(f"Client {self.client_id} - Epoch {epoch+1}: "
                              f"Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
                        
            self.save_model(epoch_acc)
        
        # Save training plots after all epochs
        self.save_training_plots()
        print(f"🎉 Client {self.client_id} completed training")


if __name__ == "__main__":
    # Example: Sequential client execution
    for client_id in range(SystemConfig.CLIENT_NUM):
        client = FLClient(client_id)
        client.train(num_epochs=TrainingConfig.NUM_EPOCHS)
        torch.cuda.empty_cache()
        print(f"🛑 Client {client.client_id} finished")