# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
import sys
import traceback

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
        checkpoint_dir: str,
        start_epoch: int = 0,
        best_val_loss: float = float('inf')
):
    writer = SummaryWriter(log_dir='runs')

    print(f"Starting training on device: {device}")
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")

    # Add skip counter
    skip_counter = 0
    max_skip_ratio = 0.5  # Maximum ratio of skipped batches before raising warning

    for epoch in range(start_epoch, num_epochs):
        # Training phase
        cnn.train()
        vit.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_skip_counter = 0

        for batch_idx, (videos, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}')):
            try:
                # Check if batch is all zeros
                if videos.sum() == 0:
                    epoch_skip_counter += 1
                    if epoch_skip_counter / len(train_loader) > max_skip_ratio:
                        print("\nWarning: Too many batches being skipped. Check your dataset!")
                    continue

                videos = videos.to(device)
                labels = labels.float().to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                temporal_features = cnn(videos)
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

            except Exception as e:
                print(f"\nError in training batch {batch_idx}: {str(e)}")
                continue

        if train_total == 0:
            print("\nError: No valid training samples in this epoch!")
            continue

        avg_train_loss = train_loss / (len(train_loader) - epoch_skip_counter)
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

        # Save backup checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            backup_checkpoint = {
                'cnn_state_dict': cnn.state_dict(),
                'vit_state_dict': vit.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
            }
            backup_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save(backup_checkpoint, backup_path)
            print(f"Saved backup checkpoint to {backup_path}")

    writer.close()


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

        # Initialize loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(
            list(cnn.parameters()) + list(vit.parameters()),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Create checkpoint directory if it doesn't exist
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

        # Check for existing checkpoint
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
        start_epoch = 0
        best_val_loss = float('inf')

        if os.path.exists(checkpoint_path):
            print("Loading checkpoint...")
            checkpoint = torch.load(checkpoint_path)
            cnn.load_state_dict(checkpoint['cnn_state_dict'])
            vit.load_state_dict(checkpoint['vit_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['val_loss']
            print(f"Resuming from epoch {start_epoch}")
        else:
            print("No checkpoint found, starting from scratch")

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
            checkpoint_dir=config.CHECKPOINT_DIR,
            start_epoch=start_epoch,
            best_val_loss=best_val_loss
        )

    except Exception as e:
        print(f"Error during training setup: {str(e)}")
        print(traceback.format_exc())


if __name__ == '__main__':
    main()