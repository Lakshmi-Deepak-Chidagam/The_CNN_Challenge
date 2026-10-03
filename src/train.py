import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import ConvNeXt_Tiny_Weights


SEED = 0
NUM_CLASSES = 16
INPUT_SIZE = 224
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 10


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Improve reproducibility where supported.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transform():
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def create_loaders(data_root):
    transform = get_transform()

    dataset = datasets.ImageFolder(
        data_root,
        transform=transform
    )

    if len(dataset.classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, found {len(dataset.classes)}."
        )

    # Reproduce the 80/20 split using seed 0.
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(
        len(dataset),
        generator=generator
    ).tolist()

    train_size = int(0.8 * len(dataset))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, dataset.classes


def create_model():
    weights = ConvNeXt_Tiny_Weights.DEFAULT

    model = models.convnext_tiny(weights=weights)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(
        in_features,
        NUM_CLASSES
    )

    return model


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    loss = total_loss / total
    accuracy = correct / total

    return loss, accuracy


def train(data_root, output_path):
    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    train_loader, val_loader, class_names = create_loaders(
        data_root
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Classes: {len(class_names)}")

    model = create_model().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_accuracy = 0.0
    best_epoch = 0

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

        train_loss = running_loss / total

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy * 100:.2f}%"
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_name": "convnext_tiny",
                "num_classes": NUM_CLASSES,
                "class_names": class_names,
                "input_size": INPUT_SIZE,
                "best_val_accuracy": best_val_accuracy,
                "best_epoch": best_epoch,
                "seed": SEED
            }

            torch.save(checkpoint, output_path)

            print(
                f"Saved new best checkpoint "
                f"({best_val_accuracy * 100:.2f}%)"
            )

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )
    print(f"Checkpoint: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to ImageFolder-compatible training dataset."
    )

    parser.add_argument(
        "--output",
        default="checkpoints/convnext_tiny_best.pth",
        help="Output checkpoint path."
    )

    args = parser.parse_args()

    train(
        args.data_root,
        args.output
    )


if __name__ == "__main__":
    main()
