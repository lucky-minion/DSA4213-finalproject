import torch
import torch.nn as nn
from transformers import BertModel

pretrain_model_path = '../bert-base-cased'
# 服务器端的 Part1 模型
class ModelPart1(nn.Module):
    def __init__(self):
        super(ModelPart1, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path)
        self.encoder_layers = self.bert.encoder.layer[1:-1]  # 第 1 层到倒数第 2 层

    def forward(self, hidden_states, attention_mask):
        attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float)
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        return hidden_states

# 客户端的 Part0 和 Part2
class ModelPart0(nn.Module):
    def __init__(self):
        super(ModelPart0, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path)
        self.embeddings = self.bert.embeddings
        self.encoder_layer_0 = self.bert.encoder.layer[0]

    def forward(self, input_ids, attention_mask):
        x = self.embeddings(input_ids)
        attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float)
        x = self.encoder_layer_0(x, attention_mask=attention_mask)[0]
        return x

class ModelPart2(nn.Module):
    def __init__(self, num_classes):
        super(ModelPart2, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path)
        self.encoder_last = self.bert.encoder.layer[-1]
        self.pooler = self.bert.pooler
        #self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)  # 修改输出维度
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, hidden_states, attention_mask):
        attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float)
        hidden_states = self.encoder_last(hidden_states, attention_mask=attention_mask)[0]
        pooled_output = self.pooler(hidden_states)
        return self.classifier(pooled_output)