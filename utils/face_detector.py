import cv2
import numpy as np
from mtcnn import MTCNN
from typing import List, Tuple, Optional
import logging


class FaceDetector:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.detector = MTCNN()
        self.target_size = target_size
        # Configure logging to be less verbose
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

    def detect_and_crop_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and crop the largest face in the frame.
        Returns None if no face is detected.
        """
        try:
            # Detect faces
            faces = self.detector.detect_faces(frame)

            if not faces:
                return None

            # Get the largest face
            largest_face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
            box = largest_face['box']

            # Add margin
            margin = 0.2
            x, y, w, h = box
            x = max(0, x - int(w * margin))
            y = max(0, y - int(h * margin))
            w = min(frame.shape[1] - x, int(w * (1 + 2 * margin)))
            h = min(frame.shape[0] - y, int(h * (1 + 2 * margin)))

            # Crop and resize
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, self.target_size)

            return face

        except Exception as e:
            print(f"Warning: Error in face detection: {str(e)}")
            return None

    def process_video(self, video_path: str, sequence_length: int) -> Optional[np.ndarray]:
        """
        Process video file and return sequences of face crops.
        Returns None if no faces could be detected.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        consecutive_failures = 0
        max_consecutive_failures = 10

        while len(frames) < sequence_length:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect and crop face
            face = self.detect_and_crop_face(frame)
            if face is not None:
                frames.append(face)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"Warning: Failed to detect faces in {max_consecutive_failures} consecutive frames")
                    break

        cap.release()

        # If we don't have enough frames but have at least one
        if frames and len(frames) < sequence_length:
            # Pad with the last valid frame
            while len(frames) < sequence_length:
                frames.append(frames[-1])
        elif not frames:
            # If no faces were detected at all, return None
            print(f"Warning: No faces detected in video: {video_path}")
            return None

        return np.array(frames)