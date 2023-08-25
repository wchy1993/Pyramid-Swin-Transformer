import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import get_dataloaders
from model import get_model, get_loss
from config import *
import torch


def train():
    model = get_model()
    criterion = get_loss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_loader, val_loader = get_dataloaders(traindir, valdir, train_batch_size, val_batch_size)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        torch.save(model.state_dict(), 'weights_epoch_{}.pth'.format(epoch + 1))
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {100. * correct / total:.2f}%')

if __name__ == "__main__":
    train()

