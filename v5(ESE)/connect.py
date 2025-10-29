"""
Network Communication Utilities with Extreme Sketch Encryption

This module provides functions for secure socket communication between clients and server
using Extreme Sketch Encryption for gradient protection.
"""

import pickle
import struct
import numpy as np
import hashlib
import torch
from tqdm import tqdm

class ExtremeSketchEncryptor:
    """Extreme Sketch Encryption for secure gradient transmission
        Client-side usage:
            # Initialize encryptor (should use same seed/size across system)
            encryptor = initialize_encryptor(sketch_size=2048, seed=42)

            # After computing gradients, replace original gradient sending with:
            encrypt_and_send_gradients(train_socket, model_part2, encryptor)

        Server-side usage:
            # Initialize encryptor (same parameters as clients)
            encryptor = initialize_encryptor(sketch_size=2048, seed=42)

            # Replace gradient receiving with:
            approx_grads = receive_and_decrypt_gradients(conn, encryptor, model_part2_template)

            # Apply decrypted gradients to model
            if approx_grads and SystemConfig.SERVER_MODEL_PART_BACKFORWARD:
                for name, param in model.named_parameters():
                    if name in approx_grads:
                        param.grad = approx_grads[name].to(device)
                optimizer.step()
    """
    
    def __init__(self, sketch_size=1000, seed=42):
        self.sketch_size = sketch_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _hash(self, name, dim):
        """Generate hash-based index and sign for sketch construction"""
        index = int(hashlib.md5(f"{name}_{dim}".encode()).hexdigest(), 16) % self.sketch_size
        sign = 1 if int(hashlib.sha1(f"{name}_{dim}".encode()).hexdigest(), 16) % 2 == 0 else -1
        return index, sign

    def encrypt_gradients(self, model):
        """Encrypt model gradients into a sketch with progress display"""
        sketch = np.zeros(self.sketch_size)
        
        total_params = sum(p.grad.numel() for p in model.parameters() if p.grad is not None)
        
        with tqdm(total=total_params, 
                 desc="Encrypting Gradients", 
                 unit="param",
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_flat = param.grad.view(-1).detach().cpu().numpy()
                    
                    for i, val in enumerate(grad_flat):
                        idx, sign = self._hash(name, i)
                        sketch[idx] += sign * val
                        pbar.update(1)  # 更新进度条
                        
            mask = self.rng.normal(0, 1, self.sketch_size)
            encrypted_sketch = sketch + mask
            
            # 添加加密后信息显示
            pbar.set_postfix({
                "sketch_size": f"{self.sketch_size}D",
                "compression": f"{total_params/sketch.size:.1f}x"
            })
            
        return encrypted_sketch, mask

    @staticmethod
    def aggregate_sketches(encrypted_sketches):
        """Aggregate multiple encrypted sketches"""
        return np.sum(encrypted_sketches, axis=0)

    @staticmethod
    def aggregate_masks(masks):
        """Aggregate multiple masks"""
        return np.sum(masks, axis=0)

    def decrypt_aggregated_sketch(self, agg_encrypted_sketch, agg_mask):
        """Decrypt aggregated sketch using aggregated mask"""
        return agg_encrypted_sketch - agg_mask

    def decode_sketch_to_grad(self, model_template, sketch):
        """Decode sketch back to approximate gradients"""
        named_grads = {}
        for name, param in model_template.named_parameters():
            if param.requires_grad:
                grad_flat = []
                for i in range(param.numel()):
                    idx, sign = self._hash(name, i)
                    grad_flat.append(sign * sketch[idx])
                grad_tensor = torch.tensor(grad_flat, dtype=torch.float32).view(param.shape)
                named_grads[name] = grad_tensor
        return named_grads

def receive_full_data(conn):
    """Receive complete data with length prefix
    
    Args:
        conn (socket): Connection socket
        
    Returns:
        object: Unpickled data or None on error
    """
    try:
        # Receive 4-byte length prefix
        raw_msglen = recvall(conn, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('!I', raw_msglen)[0]
        
        # Receive actual data
        data = recvall(conn, msglen)
        return pickle.loads(data)
    except (ConnectionResetError, struct.error):
        return None

def send_full_data(conn, data):
    """Send data with length prefix
    
    Args:
        conn (socket): Connection socket
        data (object): Data to send
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        data = pickle.dumps(data)
        conn.sendall(struct.pack('!I', len(data)) + data)
    except (ConnectionResetError, BrokenPipeError):
        return False
    return True

def recvall(conn, n):
    """Receive exactly n bytes
    
    Args:
        conn (socket): Connection socket
        n (int): Number of bytes to receive
        
    Returns:
        bytes: Received data or None on error
    """
    data = bytearray()
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)

def encrypt_and_send_gradients(conn, model, encryptor):
    """Encrypt model gradients and send to server
    
    Args:
        conn (socket): Connection socket
        model (nn.Module): Model containing gradients to encrypt
        encryptor (ExtremeSketchEncryptor): ESE encryptor instance
    """
    encrypted_sketch, mask = encryptor.encrypt_gradients(model)
    send_full_data(conn, {
        'encrypted_sketch': encrypted_sketch,
        'mask': mask
    })

def receive_and_decrypt_gradients(conn, encryptor, model_template):
    """Receive and decrypt aggregated gradients from clients
    
    Args:
        conn (socket): Connection socket
        encryptor (ExtremeSketchEncryptor): ESE encryptor instance
        model_template (nn.Module): Model structure template
        
    Returns:
        dict: Decrypted approximate gradients
    """
    data = receive_full_data(conn)
    if not data or 'encrypted_sketch' not in data:
        return None
        
    encrypted_sketch = data['encrypted_sketch']
    mask = data['mask']
    
    # In real FL, server would aggregate multiple sketches first
    # Here we demonstrate single client case
    decrypted_sketch = encryptor.decrypt_aggregated_sketch(encrypted_sketch, mask)
    return encryptor.decode_sketch_to_grad(model_template, decrypted_sketch)

def initialize_encryptor(sketch_size=2048, seed=None):
    """Initialize ESE encryptor with given parameters
    
    Args:
        sketch_size (int): Size of sketch vector
        seed (int): Random seed for reproducibility
        
    Returns:
        ExtremeSketchEncryptor: Initialized encryptor instance
    """
    if seed is None:
        seed = np.random.randint(0, 10000)
    return ExtremeSketchEncryptor(sketch_size=sketch_size, seed=seed)