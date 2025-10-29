import torch
import os
import sys
from glob import glob
from collections import defaultdict
from config import SystemConfig, ModelConfig, DatasetConfig
from split_model import ModelPart0, ModelPart1, ModelPart2

class ModelAggregator:
    def __init__(self, model_part='all'):
        """
        初始化模型聚合器
        
        Args:
            model_part (str): 要聚合的模型部分 ('part0', 'part1', 'part2', 'all')
        """
        self.device = SystemConfig.DEVICE
        self.model_part = model_part
        self.client_models_part0 = []
        self.client_models_part1 = []
        self.client_models_part2 = []
        
        # 创建保存聚合模型的目录
        os.makedirs("./aggregated_models", exist_ok=True)
    
    def load_client_models(self):
        """加载所有客户端保存的模型"""
        # 加载part0模型
        part0_paths = glob("./client_saved_models/modelpart0/*.pt")
        for path in part0_paths:
            model = ModelPart0().to(self.device)
            model.load_state_dict(torch.load(path, map_location=self.device))
            self.client_models_part0.append(model)
        
        # 加载part2模型
        part2_paths = glob("./client_saved_models/modelpart2/*.pt")
        for path in part2_paths:
            model = ModelPart2(DatasetConfig.DATASET_NUM_CLASSES).to(self.device)
            model.load_state_dict(torch.load(path, map_location=self.device))
            self.client_models_part2.append(model)
        
        # 加载part1模型 (服务器端)
        part1_paths = glob("./server_saved_models/*.pt")
        for path in part1_paths:
            model = ModelPart1().to(self.device)
            model.load_state_dict(torch.load(path, map_location=self.device))
            self.client_models_part1.append(model)
        
        print(f"Loaded {len(self.client_models_part0)} part0 models")
        print(f"Loaded {len(self.client_models_part1)} part1 models")
        print(f"Loaded {len(self.client_models_part2)} part2 models")
    
    def _get_client_weights(self):
        """
        根据客户端性能因子计算权重
        假设性能因子与客户端ID相关 (client_id+1)/CLIENT_NUM
        """
        weights = [(i+1)/SystemConfig.CLIENT_NUM for i in range(SystemConfig.CLIENT_NUM)]
        # 归一化权重
        total = sum(weights)
        return [w/total for w in weights]
    
    def average_aggregate(self):
        """平均聚合所有客户端的模型"""
        print("\nPerforming average aggregation...")
        
        if self.model_part in ['part0', 'all'] and self.client_models_part0:
            # 使用第一个模型的state_dict作为基础
            avg_state_dict = self.client_models_part0[0].state_dict()
            
            # 累加所有模型参数
            for model in self.client_models_part0[1:]:
                model_state_dict = model.state_dict()
                for name, param in model_state_dict.items():
                    avg_state_dict[name] += param.data
            
            # 计算平均值
            for name in avg_state_dict:
                avg_state_dict[name] = avg_state_dict[name] / len(self.client_models_part0)
            
            # 创建新模型并保存
            avg_part0 = ModelPart0().to(self.device)
            avg_part0.load_state_dict(avg_state_dict)
            torch.save(avg_part0.state_dict(), "./aggregated_models/avg_part0.pt")
            print("Saved averaged part0 model")
        
        if self.model_part in ['part1', 'all'] and self.client_models_part1:
            avg_state_dict = self.client_models_part1[0].state_dict()
            
            for model in self.client_models_part1[1:]:
                model_state_dict = model.state_dict()
                for name, param in model_state_dict.items():
                    avg_state_dict[name] += param.data
            
            for name in avg_state_dict:
                avg_state_dict[name] = avg_state_dict[name] / len(self.client_models_part1)
            
            avg_part1 = ModelPart1().to(self.device)
            avg_part1.load_state_dict(avg_state_dict)
            torch.save(avg_part1.state_dict(), "./aggregated_models/avg_part1.pt")
            print("Saved averaged part1 model")
        
        if self.model_part in ['part2', 'all'] and self.client_models_part2:
            avg_state_dict = self.client_models_part2[0].state_dict()
            
            for model in self.client_models_part2[1:]:
                model_state_dict = model.state_dict()
                for name, param in model_state_dict.items():
                    avg_state_dict[name] += param.data
            
            for name in avg_state_dict:
                avg_state_dict[name] = avg_state_dict[name] / len(self.client_models_part2)
            
            avg_part2 = ModelPart2(DatasetConfig.DATASET_NUM_CLASSES).to(self.device)
            avg_part2.load_state_dict(avg_state_dict)
            torch.save(avg_part2.state_dict(), "./aggregated_models/avg_part2.pt")
            print("Saved averaged part2 model")

    def weighted_aggregate(self):
        """基于性能因子的加权聚合"""
        print("\nPerforming weighted aggregation based on client performance...")
        weights = self._get_client_weights()
        
        if self.model_part in ['part0', 'all'] and self.client_models_part0:
            # 使用第一个模型的state_dict作为基础，并乘以权重
            weighted_state_dict = self.client_models_part0[0].state_dict()
            for name in weighted_state_dict:
                weighted_state_dict[name] = weighted_state_dict[name] * weights[0]
            
            # 累加所有模型参数
            for i, model in enumerate(self.client_models_part0[1:], 1):
                model_state_dict = model.state_dict()
                for name, param in model_state_dict.items():
                    weighted_state_dict[name] += param.data * weights[i]
            
            # 创建新模型并保存
            weighted_part0 = ModelPart0().to(self.device)
            weighted_part0.load_state_dict(weighted_state_dict)
            torch.save(weighted_part0.state_dict(), "./aggregated_models/weighted_part0.pt")
            print("Saved weighted part0 model")
        
        if self.model_part in ['part1', 'all'] and self.client_models_part1:
            weighted_state_dict = self.client_models_part1[0].state_dict()
            for name in weighted_state_dict:
                weighted_state_dict[name] = weighted_state_dict[name] * weights[0]
            
            for i, model in enumerate(self.client_models_part1[1:], 1):
                model_state_dict = model.state_dict()
                for name, param in model_state_dict.items():
                    weighted_state_dict[name] += param.data * weights[i]
            
            weighted_part1 = ModelPart1().to(self.device)
            weighted_part1.load_state_dict(weighted_state_dict)
            torch.save(weighted_part1.state_dict(), "./aggregated_models/weighted_part1.pt")
            print("Saved weighted part1 model")
        
        if self.model_part in ['part2', 'all'] and self.client_models_part2:
            weighted_state_dict = self.client_models_part2[0].state_dict()
            for name in weighted_state_dict:
                weighted_state_dict[name] = weighted_state_dict[name] * weights[0]
            
            for i, model in enumerate(self.client_models_part2[1:], 1):
                model_state_dict = model.state_dict()
                for name, param in model_state_dict.items():
                    weighted_state_dict[name] += param.data * weights[i]
            
            weighted_part2 = ModelPart2(DatasetConfig.DATASET_NUM_CLASSES).to(self.device)
            weighted_part2.load_state_dict(weighted_state_dict)
            torch.save(weighted_part2.state_dict(), "./aggregated_models/weighted_part2.pt")
            print("Saved weighted part2 model")
    
    def combine_aggregated_models(self):
        """组合聚合后的模型部分为完整模型"""
        # 加载平均聚合的模型
        avg_part0 = ModelPart0().to(self.device)
        avg_part0.load_state_dict(torch.load("./aggregated_models/avg_part0.pt", map_location=self.device))
        
        avg_part1 = ModelPart1().to(self.device)
        avg_part1.load_state_dict(torch.load("./aggregated_models/avg_part1.pt", map_location=self.device))
        
        avg_part2 = ModelPart2(DatasetConfig.DATASET_NUM_CLASSES).to(self.device)
        avg_part2.load_state_dict(torch.load("./aggregated_models/avg_part2.pt", map_location=self.device))
        
        # 保存完整模型
        full_model = {
            'part0': avg_part0.state_dict(),
            'part1': avg_part1.state_dict(),
            'part2': avg_part2.state_dict()
        }
        torch.save(full_model, "./aggregated_models/full_avg_model.pt")
        print("Saved full averaged model")
        
        # 加载加权聚合的模型
        weighted_part0 = ModelPart0().to(self.device)
        weighted_part0.load_state_dict(torch.load("./aggregated_models/weighted_part0.pt", map_location=self.device))
        
        weighted_part1 = ModelPart1().to(self.device)
        weighted_part1.load_state_dict(torch.load("./aggregated_models/weighted_part1.pt", map_location=self.device))
        
        weighted_part2 = ModelPart2(DatasetConfig.DATASET_NUM_CLASSES).to(self.device)
        weighted_part2.load_state_dict(torch.load("./aggregated_models/weighted_part2.pt", map_location=self.device))
        
        # 保存完整模型
        full_model = {
            'part0': weighted_part0.state_dict(),
            'part1': weighted_part1.state_dict(),
            'part2': weighted_part2.state_dict()
        }
        torch.save(full_model, "./aggregated_models/full_weighted_model.pt")
        print("Saved full weighted model")

if __name__ == "__main__":
    # 示例用法
    aggregator = ModelAggregator(model_part='all')
    aggregator.load_client_models()
    
    # 执行两种聚合方法
    aggregator.average_aggregate()
    aggregator.weighted_aggregate()
    
    # 组合聚合后的模型
    # aggregator.combine_aggregated_models()