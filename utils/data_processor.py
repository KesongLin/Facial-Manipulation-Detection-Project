import torch
from torch.utils.data import Dataset
import os
from typing import List, Tuple
import numpy as np
from .face_detector import FaceDetector
import cv2


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir: str, sequence_length: int = 16):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.face_detector = FaceDetector()
        self.target_size = (224, 224)
        self.samples = self._load_dataset()
        self.valid_samples = []

        print(f"Found {len(self.samples)} total videos")
        self._validate_samples()
        print(f"After validation: {len(self.valid_samples)} valid videos")

        if len(self.valid_samples) == 0:
            raise RuntimeError("No valid videos found in the dataset!")

    def _generate_dummy_frames(self) -> np.ndarray:
        """Generate dummy frames when face detection fails."""
        return np.zeros((self.sequence_length, self.target_size[0], self.target_size[1], 3), dtype=np.float32)

    def _load_dataset(self) -> List[Tuple[str, int]]:
        samples = []

        # Real videos
        real_dir = os.path.join(self.root_dir, "real")
        if os.path.exists(real_dir):
            for video in os.listdir(real_dir):
                if video.endswith((".mp4", ".avi")):
                    samples.append((os.path.join(real_dir, video), 0))

        # Fake videos
        fake_dir = os.path.join(self.root_dir, "fake")
        if os.path.exists(fake_dir):
            for video in os.listdir(fake_dir):
                if video.endswith((".mp4", ".avi")):
                    samples.append((os.path.join(fake_dir, video), 1))

        return samples

    def _validate_samples(self):
        """Pre-validate all samples to ensure they contain detectable faces."""
        for video_path, label in self.samples:
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Warning: Cannot open video file {video_path}")
                    continue

                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: Cannot read frames from {video_path}")
                    continue

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Try face detection on first frame
                face = self.face_detector.detect_and_crop_face(frame)
                if face is not None:
                    self.valid_samples.append((video_path, label))
                else:
                    print(f"Warning: No faces detected in first frame of {video_path}")

                cap.release()

            except Exception as e:
                print(f"Error validating {video_path}: {str(e)}")
                continue

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.valid_samples[idx]

        try:
            # Try to process video frames
            frames = self.face_detector.process_video(video_path, self.sequence_length)

            # If face detection failed, use dummy frames
            if frames is None:
                print(f"Warning: Face detection failed for {video_path}, using dummy frames")
                frames = self._generate_dummy_frames()

            # Ensure frames is a numpy array
            frames = np.array(frames, dtype=np.float32)

            # Double-check the shape
            if frames.shape != (self.sequence_length, self.target_size[0], self.target_size[1], 3):
                print(f"Warning: Unexpected frame shape for {video_path}, using dummy frames")
                frames = self._generate_dummy_frames()

            # Convert to tensor and normalize
            frames = torch.FloatTensor(frames)
            frames = frames.permute(3, 0, 1, 2)  # [C, T, H, W]
            frames = frames / 255.0

            return frames, label

        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            # Return dummy data
            frames = self._generate_dummy_frames()
            frames = torch.FloatTensor(frames)
            frames = frames.permute(3, 0, 1, 2)
            frames = frames / 255.0
            return frames, label

    def get_sample_path(self, idx: int) -> str:
        """Get the file path for a sample."""
        return self.valid_samples[idx][0]