import argparse
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def create_model(num_classes):
    model = models.convnext_tiny(weights=None)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(
        in_features,
        num_classes
    )

    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to ImageFolder-compatible test dataset."
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained checkpoint."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )

    num_classes = checkpoint.get(
        "num_classes",
        16
    )

    input_size = checkpoint.get(
        "input_size",
        224
    )

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = datasets.ImageFolder(
        args.data_root,
        transform=transform
    )

    checkpoint_classes = checkpoint.get("class_names")

    if (
        checkpoint_classes is not None
        and test_dataset.classes != checkpoint_classes
    ):
        raise ValueError(
            "Test class ordering does not match the checkpoint."
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    model = create_model(num_classes).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    test_loss = total_loss / total
    test_accuracy = correct / total

    print("\nFinal Evaluation")
    print("=" * 50)
    print("Model: ConvNeXt-Tiny")
    print(f"Test images: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Test loss: {test_loss:.4f}")
    print(
        f"Test accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
