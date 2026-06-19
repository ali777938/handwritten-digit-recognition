import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout1(x)
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_data  = datasets.MNIST("data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)

    model = DigitCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    best_acc = 0

    for epoch in range(1, 11):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += (model(imgs).argmax(1) == labels).sum().item()
                total += len(labels)
        acc = correct / total * 100
        scheduler.step()
        print(f"Epoch {epoch}/10 | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/digit_model.pth")
            print(f"  Saved best model ({best_acc:.2f}%)")

    print(f"\nDone! Best accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train()
