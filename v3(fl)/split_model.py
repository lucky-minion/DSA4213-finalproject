"""
Split BERT Model Implementation

This module implements the split BERT model architecture for federated learning.
"""

import torch
import torch.nn as nn
from transformers import BertModel

pretrain_model_path = '../bert-base-cased'

class ModelPart1(nn.Module):
    """Server-side model part (middle layers)"""
    
    def __init__(self):
        super(ModelPart1, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path)
        self.encoder_layers = self.bert.encoder.layer[1:-1]  # Layers 1 to second last

    def forward(self, hidden_states, attention_mask):
        """Forward pass
        
        Args:
            hidden_states (torch.Tensor): Input hidden states
            attention_mask (torch.Tensor): Attention mask
            
        Returns:
            torch.Tensor: Output hidden states
        """
        attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float)
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        return hidden_states

class ModelPart0(nn.Module):
    """Client-side first part (embeddings + first layer)"""
    
    def __init__(self):
        super(ModelPart0, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path)
        self.embeddings = self.bert.embeddings
        self.encoder_layer_0 = self.bert.encoder.layer[0]

    def forward(self, input_ids, attention_mask):
        """Forward pass
        
        Args:
            input_ids (torch.Tensor): Input token ids
            attention_mask (torch.Tensor): Attention mask
            
        Returns:
            torch.Tensor: Output hidden states
        """
        x = self.embeddings(input_ids)
        attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float)
        x = self.encoder_layer_0(x, attention_mask=attention_mask)[0]
        return x

class ModelPart2(nn.Module):
    """Client-side last part (last layer + classifier)"""
    
    def __init__(self, num_classes):
        """Initialize
        
        Args:
            num_classes (int): Number of output classes
        """
        super(ModelPart2, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path)
        self.encoder_last = self.bert.encoder.layer[-1]
        self.pooler = self.bert.pooler
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, hidden_states, attention_mask):
        """Forward pass
        
        Args:
            hidden_states (torch.Tensor): Input hidden states
            attention_mask (torch.Tensor): Attention mask
            
        Returns:
            torch.Tensor: Classification logits
        """
        attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float)
        hidden_states = self.encoder_last(hidden_states, attention_mask=attention_mask)[0]
        pooled_output = self.pooler(hidden_states)
        return self.classifier(pooled_output)