import torch
import cv2
import numpy as np
from typing import Tuple
import os

from config.config import config
from models.cnn3d import CNN3D
from models.vision_transformer import VisionTransformer
from utils.face_detector import FaceDetector


class DeepfakeDetector:
    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = device if device else config.DEVICE

        # Initialize models
        self.cnn = CNN3D(
            channels=config.CNN3D_CHANNELS,
            embedding_dim=config.CNN3D_EMBEDDING_DIM
        ).to(self.device)

        self.vit = VisionTransformer(
            embedding_dim=config.CNN3D_EMBEDDING_DIM,
            num_heads=config.VIT_NUM_HEADS,
            num_layers=config.VIT_NUM_LAYERS,
            mlp_dim=config.VIT_MLP_DIM,
            dropout=config.VIT_DROPOUT
        ).to(self.device)

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.cnn.load_state_dict(checkpoint['cnn_state_dict'])
        self.vit.load_state_dict(checkpoint['vit_state_dict'])

        # Set models to evaluation mode
        self.cnn.eval()
        self.vit.eval()

        # Initialize face detector
        self.face_detector = FaceDetector()
        print("Models initialized successfully")

    def predict_video(self, video_path: str) -> Tuple[float, float]:
        """
        Predict whether a video is real or fake.
        Returns: (manipulation_score, confidence)
        """
        print(f"Processing video: {video_path}")

        # Process video frames
        frames = self.face_detector.process_video(
            video_path,
            config.SEQUENCE_LENGTH
        )

        if frames is None:
            print("Warning: No faces detected in video")
            return 0.5, 0.0  # Return neutral prediction with zero confidence

        # Prepare input tensor
        frames = torch.FloatTensor(frames)
        frames = frames.permute(3, 0, 1, 2)  # [C, T, H, W]
        frames = frames / 255.0
        frames = frames.unsqueeze(0)  # Add batch dimension
        frames = frames.to(self.device)

        # Make prediction
        with torch.no_grad():
            temporal_features = self.cnn(frames)
            output = self.vit(temporal_features)

            manipulation_score = output.item()
            confidence = abs(manipulation_score - 0.5) * 2  # Scale to 0-1

        return manipulation_score, confidence