import pickle
import struct

def receive_full_data(conn):
    """循环接收完整数据，避免数据截断"""
    data_size = struct.unpack("!I", conn.recv(4))[0]  # 先接收数据大小（4 字节）
    data = b""

    while len(data) < data_size:
        packet = conn.recv(min(4096, data_size - len(data)))  # 分块接收
        if not packet:
            break
        data += packet

    return pickle.loads(data)  # 反序列化

def send_full_data(conn, data):
    """确保完整发送数据"""
    data = pickle.dumps(data)
    conn.sendall(struct.pack("!I", len(data)) + data)  # 先发送数据大小，再发送数据

# todo: 加密算法

# todo: 解密算法