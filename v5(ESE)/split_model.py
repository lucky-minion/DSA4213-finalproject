import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from config import ModelConfig

class ModelPart0(nn.Module):
    """Client-side first part (embeddings + first layer)"""
    def __init__(self):
        super(ModelPart0, self).__init__()
        config = BertConfig.from_pretrained(ModelConfig.PRETRAIN_MODEL_PATH)
        full_model = BertModel.from_pretrained(ModelConfig.PRETRAIN_MODEL_PATH)

        self.embeddings = full_model.embeddings
        self.encoder_layer_0 = full_model.encoder.layer[0]

    def forward(self, input_ids, attention_mask):
        x = self.embeddings(input_ids)
        attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float)
        x = self.encoder_layer_0(x, attention_mask=attention_mask)[0]
        return x

class ModelPart1(nn.Module):
    """Server-side model part (middle layers)"""
    def __init__(self):
        super(ModelPart1, self).__init__()
        config = BertConfig.from_pretrained(ModelConfig.PRETRAIN_MODEL_PATH)
        full_model = BertModel.from_pretrained(ModelConfig.PRETRAIN_MODEL_PATH)

        if ModelConfig.SPLIT_LAYER_NUM > 1:
            self.encoder_layers = nn.ModuleList(
                full_model.encoder.layer[1:ModelConfig.SPLIT_LAYER_NUM]
            )
        else:
            self.encoder_layers = nn.ModuleList([])

    def forward(self, hidden_states, attention_mask):
        attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float)
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        return hidden_states

class ModelPart2(nn.Module):
    """Client-side last part (last layers + classifier)"""
    def __init__(self, num_classes):
        super(ModelPart2, self).__init__()
        config = BertConfig.from_pretrained(ModelConfig.PRETRAIN_MODEL_PATH)
        full_model = BertModel.from_pretrained(ModelConfig.PRETRAIN_MODEL_PATH)

        self.encoder_layers = nn.ModuleList(
            full_model.encoder.layer[ModelConfig.SPLIT_LAYER_NUM:]
        )
        self.pooler = full_model.pooler
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, hidden_states, attention_mask):
        attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float)
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        pooled_output = self.pooler(hidden_states)
        return self.classifier(pooled_output)
