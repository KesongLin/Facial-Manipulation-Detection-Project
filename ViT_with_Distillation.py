import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from transformers import DeiTForImageClassificationWithTeacher
from Feature_Extraction import teacher_model, FaceImageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(teacher_model.parameters(), lr=1e-4)


vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalization as per ViT requirements
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

train_dataset_vit = FaceImageDataset(X_train_paths, y_train, transform=vit_transform)
val_dataset_vit = FaceImageDataset(X_val_paths, y_val, transform=vit_transform)

train_loader_vit = DataLoader(train_dataset_vit, batch_size=32, shuffle=True, num_workers=4)
val_loader_vit = DataLoader(val_dataset_vit, batch_size=32, shuffle=False, num_workers=4)


model_vit = DeiTForImageClassificationWithTeacher.from_pretrained(
    'facebook/deit-base-distilled-patch16-224', num_labels=2)
model_vit.to(device)


def distillation_loss(student_outputs, labels, teacher_outputs, alpha=0.5, temperature=2.0):
    # Cross-entropy loss between student predictions and true labels
    ce_loss = nn.CrossEntropyLoss()(student_outputs.logits, labels)

    # KL divergence between student and teacher outputs
    kl_loss = nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(student_outputs.logits / temperature, dim=1),
        nn.functional.softmax(teacher_outputs / temperature, dim=1)
    ) * (temperature ** 2)

    # Combine losses
    loss = alpha * ce_loss + (1 - alpha) * kl_loss
    return loss


optimizer = torch.optim.Adam(model_vit.parameters(), lr=1e-5)
num_epochs = 5
temperature = 2.0
alpha = 0.5  # Weight for the classification loss

for epoch in range(num_epochs):
    model_vit.train()
    teacher_model.eval()
    running_loss = 0.0
    for images, labels in train_loader_vit:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Teacher outputs
        with torch.no_grad():
            teacher_outputs = teacher_model(images)

        # Student outputs
        student_outputs = model_vit(images)

        # Compute distillation loss
        loss = distillation_loss(student_outputs, labels, teacher_outputs, alpha, temperature)

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader_vit.dataset)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # Validation
    model_vit.eval()
    val_running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader_vit:
            images = images.to(device)
            labels = labels.to(device)

            # Teacher outputs
            teacher_outputs = teacher_model(images)

            # Student outputs
            student_outputs = model_vit(images)

            # Compute distillation loss
            loss = distillation_loss(student_outputs, labels, teacher_outputs, alpha, temperature)
            val_running_loss += loss.item() * images.size(0)

            # Predictions
            _, predicted = torch.max(student_outputs.logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = val_running_loss / len(val_loader_vit.dataset)
    val_accuracy = correct / total
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')


