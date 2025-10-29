"""
Network Communication Utilities

This module provides functions for socket communication between clients and server.
"""

import pickle
import struct

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

# TODO: Add encryption/decryption functions