import pickle
import struct

def receive_full_data(conn):
    """接收完整数据（含4字节长度前缀）"""
    try:
        # 接收数据长度
        raw_msglen = recvall(conn, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('!I', raw_msglen)[0]
        
        # 接收实际数据
        data = recvall(conn, msglen)
        return pickle.loads(data)
    except (ConnectionResetError, struct.error):
        return None

def send_full_data(conn, data):
    """发送数据（添加4字节长度前缀）"""
    try:
        data = pickle.dumps(data)
        conn.sendall(struct.pack('!I', len(data)) + data)
    except (ConnectionResetError, BrokenPipeError):
        return False
    return True

def recvall(conn, n):
    """确保接收n字节数据"""
    data = bytearray()
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)

# todo: 加密算法
# todo: 解密算法