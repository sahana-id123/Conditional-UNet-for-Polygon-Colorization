import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.unet import UNet
from utils import PolygonDataset
import wandb

wandb.init(project="ayna-ml-assignment")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = PolygonDataset('dataset/training')
val_dataset = PolygonDataset('dataset/validation')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

model = UNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for inputs, targets, colors in train_loader:
        inputs, targets, colors = inputs.to(device), targets.to(device), colors.to(device)
        outputs = model(inputs, colors)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    wandb.log({'Train Loss': total_loss / len(train_loader), 'Epoch': epoch})
    print(f"Epoch {epoch}: Loss = {total_loss / len(train_loader)}")

torch.save(model.state_dict(), "colored_polygon_model.pth")
