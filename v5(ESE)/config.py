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
    CLIENT_NUM = 6  # Total number of clients
    CLIENT_MODEL_PART_BACKFORWARD = True  # Whether clients do local backpropagation
    SERVER_MODEL_PART_BACKFORWARD = False
    
    # Server settings
    SERVER_SAVE_DIR = "./server_saved_models"
    CLIENT_SAVE_DIR = "./client_saved_models"
    FULL_MODEL_SAVE_DIR = "./full_bert_models"
    PRELOAD_TESTSET = False  # Whether to preload test dataset

# ========== Model Configuration ==========
class ModelConfig:
    # Pretrained model path
    PRETRAIN_MODEL_PATH = "../bert-base-cased"
    
    # Model architecture
    HIDDEN_SIZE = 768  # BERT base hidden size
    INTERMEDIATE_SIZE = 512  # Classifier intermediate layer size
    DROPOUT_PROB = 0.1  # Dropout probability
    
    # Split points for federated learning
    SPLIT_LAYER_NUM = 10
    LAST_LAYER_NUM = 11  # Last layer of original model
    CLIENT_PART0_LAYERS = [0]  # Layers 0 stays on client
    SERVER_PART1_LAYERS = list(range(1, SPLIT_LAYER_NUM))  # Layers 1-10 on server
    CLIENT_PART2_LAYERS = list(range(SPLIT_LAYER_NUM, LAST_LAYER_NUM + 1))
    
    

# ========== Training Configuration ==========
class TrainingConfig:
    # General training parameters
    NUM_EPOCHS = 25
    LEARNING_RATE = 5e-6
    
    # Batch sizes
    TRAIN_BATCH_SIZE = 50
    TEST_BATCH_SIZE = 250
    
    # Dataset allocation
    MIN_DATA_FACTOR = 0.5  # Minimum data percentage for clients
    MAX_DATA_FACTOR = 1.0  # Maximum data percentage for clients
    
    AGGREGATION_INTERVAL = 1

# ========== Dataset Configuration ==========
class DatasetConfig:
    # Dataset paths
    TREC_PATH = '../trec'
    AG_NEWS_PATH = '../ag_news_dataset/'
    YAHOO_PATH = '../yahoo_answers_topics'
    EMOTION_PATH = '../emotion'
    BANKING77_PATH = '../banking77'

    # Preprocessing
    TREC_MAX_LENGTH = 32  # Short sequences for TREC
    NEWS_MAX_LENGTH = 128
    YAHOO_MAX_LENGTH = 128
    EMOTION_MAX_LENGTH = 64
    BANKING77_MAX_LENGTH = 64
    
    # Classification
    TREC_COARSE_NUM_CLASSES = 6  # Coarse-grained classes for TREC
    TREC_FINE_NUM_CLASSES = 50   # Fine-grained classes for TREC
    AG_NEWS_NUM_CLASSES = 4
    YAHOO_NUM_CLASSES = 10
    EMOTION_NUM_CLASSES = 6
    BANKING77_NUM_CLASSES = 77
    
    # label name
    TREC_COARSE_LABEL_NAME = "coarse_label"
    TREC_FINE_LABEL_NAME = "fine_label"
    NEWS_LABLE_NAME = "label"
    YAHOO_LABEL_NAME = "topic"
    EMOTION_LABLE_NAME = "label"
    BANKING77_LABLE_NAME = "label"

    # trec dataset
    DATASET_PATH = EMOTION_PATH
    MAX_LENGTH = EMOTION_MAX_LENGTH
    # DATASET_NUM_CLASSES = TREC_COARSE_NUM_CLASSES
    # LABEL_NAME = "coarse_label"
    DATASET_NUM_CLASSES = EMOTION_NUM_CLASSES
    LABEL_NAME = EMOTION_LABLE_NAME
