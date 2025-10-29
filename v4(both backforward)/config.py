"""
Global Configuration for Federated Learning System

This file contains all shared parameters and configurations for:
- Server
- Clients
- Model architecture
- Training parameters
- Dataset settings
- Network settings
"""

import torch

# ========== System Configuration ==========
class SystemConfig:
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Base ports
    CONTROL_PORT = 50010  # Base port for control communication
    
    # Client settings
    CLIENT_NUM = 1  # Total number of clients
    CLIENT_MODEL_PART_BACKFORWARD = True  # Whether clients do local backpropagation
    
    # Server settings
    SERVER_SAVE_DIR = "./server_saved_models"
    CLIENT_SAVE_DIR = "./client_saved_models"
    FULL_MODEL_SAVE_DIR = "./full_bert_models"
    PRELOAD_TESTSET = True  # Whether to preload test dataset

# ========== Model Configuration ==========
class ModelConfig:
    # Pretrained model path
    PRETRAIN_MODEL_PATH = '../bert-base-cased'
    
    # Model architecture
    HIDDEN_SIZE = 768  # BERT base hidden size
    INTERMEDIATE_SIZE = 512  # Classifier intermediate layer size
    DROPOUT_PROB = 0.1  # Dropout probability
    
    # Split points for federated learning
    CLIENT_PART_LAYERS = [0]  # Layers 0 stays on client
    SERVER_PART_LAYERS = list(range(1, 11))  # Layers 1-10 on server
    LAST_LAYER = 11  # Last layer on client

# ========== Training Configuration ==========
class TrainingConfig:
    # General training parameters
    NUM_EPOCHS = 10
    LEARNING_RATE = 5e-5
    
    # Batch sizes
    TRAIN_BATCH_SIZE = 200
    TEST_BATCH_SIZE = 250
    
    # Dataset allocation
    MIN_DATA_FACTOR = 0.5  # Minimum data percentage for clients
    MAX_DATA_FACTOR = 1.0  # Maximum data percentage for clients

# ========== Dataset Configuration ==========
class DatasetConfig:
    # Dataset paths
    TREC_PATH = '../trec'
    AG_NEWS_PATH = '../ag_news_dataset'
    YAHOO_PATH = '../yahoo_answers_topics'

    # Preprocessing
    TREC_MAX_LENGTH = 32  # Short sequences for TREC
    NEWS_MAX_LENGTH = 128
    YAHOO_MAX_LENGTH = 128
    
    # Classification
    TREC_NUM_CLASSES = 6  # Coarse-grained classes for TREC
    AG_NEWS_NUM_CLASSES = 4
    YAHOO_NUM_CLASSES = 10

    # trec dataset
    DATASET_PATH = '../trec'
    MAX_LENGTH = 32
    DATASET_NUM_CLASSES = 6

    # # ag_news dataset
    # DATASET_PATH = '../ag_news_dataset'
    # MAX_LENGTH = 128
    # DATASET_NUM_CLASSES = 4

    # # yahoo answers dataset
    # DATASET_PATH = '../ag_news_dataset'
    # MAX_LENGTH = 128
    # DATASET_NUM_CLASSES = 10