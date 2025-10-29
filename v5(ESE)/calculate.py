import torch
import numpy as np
import pickle
from connect import ExtremeSketchEncryptor
from split_model import ModelPart2
from config import SystemConfig, ModelConfig, DatasetConfig

def simulate_communication_volumes():
    """模拟测试实际数据传输量"""
    # 初始化模型和加密器
    model_part2 = ModelPart2(num_classes=DatasetConfig.DATASET_NUM_CLASSES).to(SystemConfig.DEVICE)
    encryptor = ExtremeSketchEncryptor(sketch_size=2048)
    # 生成模拟梯度
    for name, param in model_part2.named_parameters():
        if param.requires_grad:
            param.grad = torch.randn_like(param) * 0.01  # 模拟梯度值
    
    # 测试原始Part2梯度大小
    original_grad_size = sum(p.grad.numel() * 4 for p in model_part2.parameters() if p.grad is not None) / (1024**2)  # MB
    
    # 测试ESE加密后数据大小
    encrypted_sketch, mask = encryptor.encrypt_gradients(model_part2)
    encrypted_size = (encrypted_sketch.nbytes + mask.nbytes) / (1024**2)  # MB
    
    # 测试实际传输数据包大小（含协议头）
    mock_data = {
        'encrypted_sketch': encrypted_sketch,
        'mask': mask,
        'client_id': 0,
        'timestamp': 1234567890
    }
    serialized_data = pickle.dumps(mock_data)
    actual_transfer_size = len(serialized_data) / (1024**2)  # MB
    
    # 计算压缩率
    compression_ratio = original_grad_size / actual_transfer_size
    
    print("\n===== 实际通信量测试结果 =====")
    print(f"原始Part2梯度大小: {original_grad_size:.2f} MB")
    print(f"ESE加密后数据大小: {encrypted_size:.4f} MB")
    print(f"实际传输数据包大小: {actual_transfer_size:.4f} MB (含协议头)")
    print(f"有效压缩比: {compression_ratio:.1f}:1")
    
    # 多客户端场景模拟
    print("\n----- 多客户端场景模拟 -----")
    clients = 6
    print(f"{clients}客户端总通信量:")
    print(f"原始梯度: {original_grad_size * clients:.2f} MB")
    print(f"ESE加密后: {actual_transfer_size * clients:.4f} MB")

def calculate_flops_full_model(L=32, H=768, V=30522, N=12, include_backprop=True):
    """
    计算完整BERT模型的总FLOPs（含前向+反向传播）
    Args:
        include_backprop: 是否包含反向传播计算量
    """
    # 前向传播
    flops_embed = 2 * L * H * V + 2 * L * H
    flops_attn = 4 * L**2 * H + 8 * L * H**2
    flops_ffn = 16 * L * H**2
    flops_classifier = 2 * H * (512 + 50)
    
    # 反向传播计算量 ≈ 2倍前向传播 (根据Pytorch官方FLOPs估算规则)
    backprop_factor = 2 if include_backprop else 1
    
    total_flops = (flops_embed + N*(flops_attn + flops_ffn) + flops_classifier) * backprop_factor
    
    breakdown = {
        "embedding": flops_embed * backprop_factor,
        "transformer_layers": N * (flops_attn + flops_ffn) * backprop_factor,
        "classifier": flops_classifier * backprop_factor,
        "backprop": "enabled" if include_backprop else "disabled"
    }
    return total_flops, breakdown

def calculate_flops_split_fedlearning(K=10, L=32, H=768, V=30522, N=12, k=2048):
    """
    分层联邦学习的FLOPs（含反向传播差异）
    """
    # 前向传播
    flops_client0 = 2 * L * H * V + (4 * L**2 * H + 24 * L * H**2)  # ModelPart0
    flops_server = (K - 1) * (4 * L**2 * H + 24 * L * H**2)         # ModelPart1
    flops_client2 = (N - K) * (4 * L**2 * H + 24 * L * H**2) + 2 * H * (512 + 50)  # ModelPart2
    
    # 反向传播特性：
    # - ModelPart0: 固定参数不更新 (无需反向传播)
    # - ModelPart2: 客户端需计算梯度 (2倍FLOPs)
    flops_client2 *= 2  # 仅ModelPart2需要反向传播
    
    # 加密计算
    d = L * H
    flops_encrypt = k + 2 * d
    flops_decrypt = N * k + d
    
    total_flops = flops_client0 + flops_server + flops_client2 + flops_encrypt + flops_decrypt
    
    breakdown = {
        "client_part0 (fwd only)": flops_client0,
        "server_part1 (fwd only)": flops_server,
        "client_part2 (fwd+bwd)": flops_client2,
        "grad_encryption": flops_encrypt,
        "backprop_scope": "only ModelPart2"
    }
    return total_flops, breakdown

def print_calculation_amount_comparison():
    """打印完整对比结果"""
    # 完整模型（含反向传播）
    flops_full, bd_full = calculate_flops_full_model(include_backprop=True)
    
    # 分层联邦（ModelPart2反向传播）
    flops_split, bd_split = calculate_flops_split_fedlearning()
    
    print("===== 计算量对比 =====")
    print(f"完整模型: {flops_full:.2f} FLOPS (含全量反向传播)")
    print(f"分层联邦: {flops_split:.2f} FLOPS (仅ModelPart2反向传播)")
    print(f"计算量减少: {(flops_full - flops_split)/flops_full:.1%}\n")
    
    print("----- 完整模型FLOPs分布 -----")
    for k, v in bd_full.items():
        if isinstance(v, str):
            print(f"{k:20}: {v}")
        else:
            print(f"{k:20}: {v:.2f} FLOPS")
    
    print("\n----- 分层联邦FLOPs分布 -----")
    for k, v in bd_split.items():
        if isinstance(v, str):
            print(f"{k:20}: {v}")
        else:
            print(f"{k:20}: {v:.2f} FLOPS")

if __name__ == "__main__":
    print_calculation_amount_comparison()
    #simulate_communication_volumes()  # 新增通信量测试
    
    # 添加带宽耗时计算
    print("\n===== 带宽耗时估算 =====")
    transfer_sizes = {
        '原始Part2梯度': 57.98,
        'ESE加密数据': 0.0312,
        '实测传输数据': 0.0315  # 根据simulate结果更新
    }
    
    for name, size in transfer_sizes.items():
        for bw in [10, 50, 100]:  # Mbps
            t = size * 8 / bw  # 转换为秒
            print(f"{name} @ {bw}Mbps: {t:.3f}s (单客户端)")