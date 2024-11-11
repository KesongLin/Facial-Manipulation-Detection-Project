import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils import shuffle
from Preprocessing import extract_frames, crop_faces
from ViT_with_Distillation import FaceImageDataset, vit_transform, DeiTForImageClassificationWithTeacher

test_real_videos_dir = 'data/test_real_videos/'
test_fake_videos_dir = 'data/test_fake_videos/'
test_real_frames_dir = 'data/test_real_frames/'
test_fake_frames_dir = 'data/test_fake_frames/'
test_real_faces_dir = 'data/test_real_faces/'
test_fake_faces_dir = 'data/test_fake_faces/'

for video_file in os.listdir(test_real_videos_dir):
    video_path = os.path.join(test_real_videos_dir, video_file)
    output_folder = os.path.join(test_real_frames_dir, video_file[:-4])
    extract_frames(video_path, output_folder)

for video_file in os.listdir(test_fake_videos_dir):
    video_path = os.path.join(test_fake_videos_dir, video_file)
    output_folder = os.path.join(test_fake_frames_dir, video_file[:-4])
    extract_frames(video_path, output_folder)

for frames_folder in os.listdir(test_real_frames_dir):
    input_folder = os.path.join(test_real_frames_dir, frames_folder)
    output_folder = os.path.join(test_real_faces_dir, frames_folder)
    crop_faces(input_folder, output_folder)

for frames_folder in os.listdir(test_fake_frames_dir):
    input_folder = os.path.join(test_fake_frames_dir, frames_folder)
    output_folder = os.path.join(test_fake_faces_dir, frames_folder)
    crop_faces(input_folder, output_folder)

def get_test_image_paths_and_labels(face_dir, label):
    image_paths = glob.glob(os.path.join(face_dir, '**', '*.jpg'), recursive=True)
    labels = [label] * len(image_paths)
    return image_paths, labels

test_real_image_paths, test_real_labels = get_test_image_paths_and_labels(test_real_faces_dir, 0)
test_fake_image_paths, test_fake_labels = get_test_image_paths_and_labels(test_fake_faces_dir, 1)

X_test_image_paths = test_real_image_paths + test_fake_image_paths
y_test_labels = test_real_labels + test_fake_labels
X_test_image_paths, y_test_labels = shuffle(X_test_image_paths, y_test_labels, random_state=42)

test_dataset_vit = FaceImageDataset(X_test_image_paths, y_test_labels, transform=vit_transform)
test_loader_vit = DataLoader(test_dataset_vit, batch_size=32, shuffle=False, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_vit = DeiTForImageClassificationWithTeacher.from_pretrained(
    'facebook/deit-base-distilled-patch16-224', num_labels=2)
model_vit.load_state_dict(torch.load('vit_model.pth'))
model_vit.to(device)
model_vit.eval()

all_preds = []
all_probs = []
all_labels = []
image_indices = []

with torch.no_grad():
    for idx, (images, labels) in enumerate(test_loader_vit):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model_vit(images)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        image_indices.extend(range(idx * test_loader_vit.batch_size, idx * test_loader_vit.batch_size + images.size(0)))

frame_acc = accuracy_score(all_labels, all_preds)
frame_prec = precision_score(all_labels, all_preds)
frame_rec = recall_score(all_labels, all_preds)
frame_f1 = f1_score(all_labels, all_preds)
frame_auc = roc_auc_score(all_labels, all_probs)
frame_cm = confusion_matrix(all_labels, all_preds)

print("Frame-Level Evaluation:")
print(f"Accuracy: {frame_acc:.4f}")
print(f"Precision: {frame_prec:.4f}")
print(f"Recall: {frame_rec:.4f}")
print(f"F1 Score: {frame_f1:.4f}")
print(f"ROC AUC Score: {frame_auc:.4f}")
print(f"Confusion Matrix:\n{frame_cm}")

image_to_video = {}
for idx, img_path in enumerate(test_dataset_vit.image_paths):
    video_id = os.path.basename(os.path.dirname(img_path))
    image_to_video[idx] = video_id

video_preds = {}
video_labels = {}

for idx, (prob, label) in enumerate(zip(all_probs, all_labels)):
    video_id = image_to_video[idx]
    if video_id not in video_preds:
        video_preds[video_id] = []
        video_labels[video_id] = label
    video_preds[video_id].append(prob)

final_video_preds = []
final_video_labels = []

for video_id in video_preds:
    avg_prob = np.mean(video_preds[video_id])
    final_pred = 1 if avg_prob > 0.5 else 0
    final_video_preds.append(final_pred)
    final_video_labels.append(video_labels[video_id])

video_acc = accuracy_score(final_video_labels, final_video_preds)
video_prec = precision_score(final_video_labels, final_video_preds)
video_rec = recall_score(final_video_labels, final_video_preds)
video_f1 = f1_score(final_video_labels, final_video_preds)
video_cm = confusion_matrix(final_video_labels, final_video_preds)

print("\nVideo-Level Evaluation:")
print(f"Accuracy: {video_acc:.4f}")
print(f"Precision: {video_prec:.4f}")
print(f"Recall: {video_rec:.4f}")
print(f"F1 Score: {video_f1:.4f}")
print(f"Confusion Matrix:\n{video_cm}")
