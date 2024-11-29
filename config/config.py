import os
import torch
from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Video processing
    FRAME_SIZE = 224  # Face crop size
    SEQUENCE_LENGTH = 16  # Number of frames to process at once

    # 3D CNN parameters
    CNN3D_CHANNELS = [32, 64, 128, 256]  # Reduced channel sizes
    CNN3D_KERNEL_SIZE = (3, 3, 3)
    CNN3D_EMBEDDING_DIM = 512

    # Vision Transformer parameters
    VIT_NUM_HEADS = 8
    VIT_NUM_LAYERS = 6
    VIT_HIDDEN_DIM = 512  # Match with CNN embedding dim
    VIT_MLP_DIM = 2048
    VIT_DROPOUT = 0.1

    # Training parameters
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    WEIGHT_DECAY = 1e-5

    # Paths
    DATA_DIR = os.path.join("data", "videos")
    CHECKPOINT_DIR = "checkpoints"

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


config = ModelConfig()