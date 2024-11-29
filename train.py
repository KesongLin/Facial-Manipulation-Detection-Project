import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
import sys

from config.config import config
from models.cnn3d import CNN3D
from models.vision_transformer import VisionTransformer
from utils.data_processor import DeepfakeDataset


def check_data_paths():
    """Check if data directories exist and contain files"""
    train_real = os.path.join(config.DATA_DIR, 'train', 'real')
    train_fake = os.path.join(config.DATA_DIR, 'train', 'fake')
    val_real = os.path.join(config.DATA_DIR, 'val', 'real')
    val_fake = os.path.join(config.DATA_DIR, 'val', 'fake')

    for path in [train_real, train_fake, val_real, val_fake]:
        if not os.path.exists(path):
            print(f"Error: Directory {path} does not exist!")
            return False

        files = [f for f in os.listdir(path) if f.endswith(('.mp4', '.avi'))]
        if not files:
            print(f"Error: No video files found in {path}")
            return False

    return True


def train_model(
        cnn: CNN3D,
        vit: VisionTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        num_epochs: int,
        device: str,
        checkpoint_dir: str
):
    writer = SummaryWriter(log_dir='runs')
    best_val_loss = float('inf')

    print(f"Starting training on device: {device}")
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")

    for epoch in range(num_epochs):
        # Training phase
        cnn.train()
        vit.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (videos, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}')):
            try:
                videos = videos.to(device)
                labels = labels.float().to(device)

                # Print shapes for debugging
                if batch_idx == 0:
                    print(f"\nInput shapes - Videos: {videos.shape}, Labels: {labels.shape}")

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass through 3D CNN
                temporal_features = cnn(videos)
                if batch_idx == 0:
                    print(f"Temporal features shape: {temporal_features.shape}")

                # Forward pass through Vision Transformer
                outputs = vit(temporal_features)

                # Calculate loss
                loss = criterion(outputs.squeeze(), labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Calculate accuracy
                predictions = (outputs.squeeze() > 0.5).float()
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)

                train_loss += loss.item()

                # Print batch statistics
                if (batch_idx + 1) % 10 == 0:
                    print(f"\nBatch {batch_idx + 1}/{len(train_loader)}:")
                    print(f"Current batch loss: {loss.item():.4f}")

            except Exception as e:
                print(f"Error in training batch {batch_idx}: {str(e)}")
                continue

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * train_correct / train_total

        # Validation phase
        cnn.eval()
        vit.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for videos, labels in tqdm(val_loader, desc='Validation'):
                try:
                    videos = videos.to(device)
                    labels = labels.float().to(device)

                    # Forward pass
                    temporal_features = cnn(videos)
                    outputs = vit(temporal_features)

                    # Calculate loss
                    loss = criterion(outputs.squeeze(), labels)
                    val_loss += loss.item()

                    # Calculate accuracy
                    predictions = (outputs.squeeze() > 0.5).float()
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)

                except Exception as e:
                    print(f"Error in validation: {str(e)}")
                    continue

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * val_correct / val_total

        # Log metrics
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'cnn_state_dict': cnn.state_dict(),
                'vit_state_dict': vit.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
            }
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")


def main():
    print("Checking CUDA availability...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    print("\nChecking data directories...")
    if not check_data_paths():
        print("Error with data directories. Please check the paths and try again.")
        return

    print("\nInitializing models...")
    try:
        # Create model instances
        cnn = CNN3D(
            channels=config.CNN3D_CHANNELS,
            embedding_dim=config.CNN3D_EMBEDDING_DIM
        ).to(config.DEVICE)

        vit = VisionTransformer(
            embedding_dim=config.CNN3D_EMBEDDING_DIM,
            num_heads=config.VIT_NUM_HEADS,
            num_layers=config.VIT_NUM_LAYERS,
            mlp_dim=config.VIT_MLP_DIM,
            dropout=config.VIT_DROPOUT
        ).to(config.DEVICE)

        print("Models initialized successfully")

        # Create datasets and dataloaders
        print("\nCreating datasets...")
        train_dataset = DeepfakeDataset(
            os.path.join(config.DATA_DIR, 'train'),
            config.SEQUENCE_LENGTH
        )
        val_dataset = DeepfakeDataset(
            os.path.join(config.DATA_DIR, 'val'),
            config.SEQUENCE_LENGTH
        )

        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True if config.DEVICE == "cuda" else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True if config.DEVICE == "cuda" else False
        )

        # Initialize loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(
            list(cnn.parameters()) + list(vit.parameters()),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Create checkpoint directory if it doesn't exist
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

        print("\nStarting training...")
        # Train the model
        train_model(
            cnn=cnn,
            vit=vit,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=config.NUM_EPOCHS,
            device=config.DEVICE,
            checkpoint_dir=config.CHECKPOINT_DIR
        )

    except Exception as e:
        print(f"Error during training setup: {str(e)}")
        import traceback
        print(traceback.format_exc())


if __name__ == '__main__':
    main()