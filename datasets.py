import os
import torch
from torchvision import datasets, transforms

def get_dataloaders(traindir, valdir, train_batch_size=32, val_batch_size=32, num_workers=4):
    """
    Returns the dataloaders for training and validation datasets.
    :param traindir: Directory containing the training dataset
    :param valdir: Directory containing the validation dataset
    :param train_batch_size: Batch size for training
    :param val_batch_size: Batch size for validation
    :param num_workers: Number of workers for dataloading
    :return: dataloaders for training and validation datasets
    """

    # Training data transformations
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(256),  # Adjusting the size to 224x224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Validation data transformations
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),  # Adjusting the size to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(traindir, transform=transform_train)
    val_dataset = datasets.ImageFolder(valdir, transform=transform_val)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader
