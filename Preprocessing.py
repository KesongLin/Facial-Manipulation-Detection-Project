import cv2
import os
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

def extract_frames(video_path, output_folder, frame_rate=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) // frame_rate
    count = 0
    success, image = cap.read()
    while success:
        if count % frame_rate == 0:
            filename = os.path.join(output_folder, f"frame_{count}.jpg")
            cv2.imwrite(filename, image)
        success, image = cap.read()
        count += 1
    cap.release()

mtcnn = MTCNN(keep_all=False)

def crop_faces(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        image = Image.open(img_path)
        # Detect face
        face = mtcnn(image)
        if face is not None:
            face_filename = os.path.join(output_folder, img_name)
            face = face.permute(1, 2, 0).int().numpy()
            cv2.imwrite(face_filename, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

def create_sequences(face_folder, sequence_length):
    sequences = []
    sequence = []
    images = sorted(os.listdir(face_folder))
    for img_name in images:
        img_path = os.path.join(face_folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        sequence.append(img)
        if len(sequence) == sequence_length:
            sequences.append(np.array(sequence))
            sequence = []
    return np.array(sequences)


