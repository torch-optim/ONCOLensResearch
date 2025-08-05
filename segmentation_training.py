import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary

from monai.data import Dataset
from preprocessing import CT_Transforms
from segmentation_model import SegmentationModel

import json
import tqdm

with open('Data/segmentation_training.json') as f:
    data_dict = json.load(f)

data_list = data_dict['Segmentation_Training']

CT_Dataset = Dataset(
    data=data_list,
    transform=CT_Transforms
    )

print(f'PyTorch version: {torch.__version__}.')

CT_Model = SegmentationModel(in_channels=1, num_classes=15)
print(summary(CT_Model, input_size=(5, 1, 64, 64, 64)))

device = torch.device('cpu')
print(f"Using Training Device: {device}")

loader = DataLoader(
    dataset=CT_Dataset,
    batch_size=1,
    shuffle=True,
    )

criterion = nn.CrossEntropyLoss()
optimizer = Adam(params=CT_Model.parameters(), lr=1e-4)

num_epochs = 5
CT_Model.to(device)

epoch_losses = []

for epoch in range(num_epochs):
    CT_Model.train()
    running_loss = 0.0
    for batch in tqdm.tqdm(loader):
        
        images = torch.stack([item['Image'].squeeze(0) for item in batch]).to(device)
        labels = torch.stack([item['Label'].squeeze(0) for item in batch]).squeeze(1).to(device).long()
        
        outputs = CT_Model(images)
        
        loss = criterion(outputs, labels.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
            
    avg_loss = running_loss / len(loader)
    epoch_losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')