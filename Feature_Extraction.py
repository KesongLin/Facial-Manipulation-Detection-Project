import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import glob
import os

# Define the teacher model
class TeacherModel(nn.Module):
    def __init__(self, num_classes=2):
        super(TeacherModel, self).__init__()
        self.model = models.resnet50(pretrained=True)
        # Replace the final layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class FaceImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def get_image_paths_and_labels(face_dir, label):
    image_paths = glob.glob(os.path.join(face_dir, '**', '*.jpg'), recursive=True)
    labels = [label] * len(image_paths)
    return image_paths, labels

# Real images
real_image_paths, real_labels = get_image_paths_and_labels('data/real_faces/', 0)

# Fake images
fake_image_paths, fake_labels = get_image_paths_and_labels('data/fake_faces/', 1)

# Combine and shuffle
all_image_paths = real_image_paths + fake_image_paths
all_labels = real_labels + fake_labels


all_image_paths, all_labels = shuffle(all_image_paths, all_labels, random_state=42)

# Split into training and validation sets

X_train_paths, X_val_paths, y_train, y_val = train_test_split(
    all_image_paths, all_labels, test_size=0.2, random_state=42)

teacher_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = FaceImageDataset(X_train_paths, y_train, transform=teacher_transform)
val_dataset = FaceImageDataset(X_val_paths, y_val, transform=teacher_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher_model = TeacherModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(teacher_model.parameters(), lr=1e-4)

num_epochs = 5

for epoch in range(num_epochs):
    teacher_model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = teacher_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # Validation
    teacher_model.eval()
    val_running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = teacher_model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = val_running_loss / len(val_loader.dataset)
    val_accuracy = correct / total
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')




