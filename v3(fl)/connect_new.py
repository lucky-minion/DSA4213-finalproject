"""
Network Communication Utilities with Encryption

This module provides secure communication functions between clients and server,
including TLS encryption and parameter encryption.
"""

import pickle
import struct
import ssl
from datetime import datetime
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os
import base64

# Encryption configuration
ENCRYPTION_CONFIG = {
    'tls_enabled': True,
    'tls_certfile': './certs/server.crt',
    'tls_keyfile': './certs/server.key',
    'param_encryption': True,
    'param_encryption_key': b'my_super_secret_key_123',  # In production, use proper key management
    'param_encryption_salt': b'salt_123'
}

def _setup_tls_context(server_side=False):
    """Create TLS/SSL context"""
    context = ssl.create_default_context(
        ssl.Purpose.CLIENT_AUTH if server_side else ssl.Purpose.SERVER_AUTH
    )
    if server_side:
        context.load_cert_chain(
            certfile=ENCRYPTION_CONFIG['tls_certfile'],
            keyfile=ENCRYPTION_CONFIG['tls_keyfile']
        )
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = False  # Disable for testing, enable in production
    return context

def _derive_encryption_key():
    """Derive encryption key from secret"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=ENCRYPTION_CONFIG['param_encryption_salt'],
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(ENCRYPTION_CONFIG['param_encryption_key'])

def _encrypt_data(data):
    """Encrypt data using AES-GCM"""
    key = _derive_encryption_key()
    iv = os.urandom(12)  # GCM recommended IV size
    cipher = Cipher(
        algorithms.AES(key),
        modes.GCM(iv),
        backend=default_backend()
    )
    encryptor = cipher.encryptor()
    encrypted = encryptor.update(data) + encryptor.finalize()
    return iv + encryptor.tag + encrypted

def _decrypt_data(encrypted_data):
    """Decrypt data using AES-GCM"""
    key = _derive_encryption_key()
    iv = encrypted_data[:12]
    tag = encrypted_data[12:28]
    ciphertext = encrypted_data[28:]
    
    cipher = Cipher(
        algorithms.AES(key),
        modes.GCM(iv, tag),
        backend=default_backend()
    )
    decryptor = cipher.decryptor()
    return decryptor.update(ciphertext) + decryptor.finalize()

def wrap_socket_with_tls(sock, server_side=False):
    """Wrap socket with TLS if enabled"""
    if ENCRYPTION_CONFIG['tls_enabled']:
        context = _setup_tls_context(server_side)
        return context.wrap_socket(sock, server_side=server_side)
    return sock

def receive_full_data(conn):
    """Receive complete data with length prefix and decryption"""
    try:
        # Receive 4-byte length prefix
        raw_msglen = recvall(conn, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('!I', raw_msglen)[0]
        
        # Receive actual data
        encrypted_data = recvall(conn, msglen)
        if not encrypted_data:
            return None
            
        # Decrypt if enabled
        if ENCRYPTION_CONFIG['param_encryption']:
            try:
                data = pickle.loads(_decrypt_data(encrypted_data))
            except Exception as e:
                print(f"⚠️ Decryption error: {e}")
                return None
        else:
            data = pickle.loads(encrypted_data)
            
        return data
    
    except (ConnectionResetError, struct.error) as e:
        print(f"⚠️ Receive error: {e}")
        return None

def send_full_data(conn, data):
    """Send data with length prefix and encryption"""
    try:
        # Serialize data
        serialized_data = pickle.dumps(data)
        
        # Encrypt if enabled
        if ENCRYPTION_CONFIG['param_encryption']:
            encrypted_data = _encrypt_data(serialized_data)
        else:
            encrypted_data = serialized_data
        
        # Send length prefix and data
        conn.sendall(struct.pack('!I', len(encrypted_data)))
        conn.sendall(encrypted_data)
        return True
        
    except (ConnectionResetError, BrokenPipeError, pickle.PicklingError) as e:
        print(f"⚠️ Send error: {e}")
        return False

def recvall(conn, n):
    """Receive exactly n bytes"""
    data = bytearray()
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)

def generate_self_signed_cert():
    """Generate self-signed certificate for testing (not for production)"""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    
    # Create key
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    
    # Create self-signed cert
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "My Company"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow()
    ).not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName([x509.DNSName("localhost")]),
        critical=False,
    ).sign(key, hashes.SHA256(), default_backend())
    
    # Create cert directory if not exists
    os.makedirs('./certs', exist_ok=True)
    
    # Write cert and key to files
    with open("./certs/server.crt", "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    
    with open("./certs/server.key", "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    
    print("✅ Generated self-signed certificate for testing")