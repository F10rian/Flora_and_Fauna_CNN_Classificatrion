import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
#import requests
import matplotlib.pyplot as plt
import warnings

#import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import random_split

import torchvision.models as models
import torch.nn as nn
from torchvision.datasets import ImageFolder


def main():
    # Setup
    # Identify available device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device: ", device)

    #load pretrained ResNet50 model
    model = models.resnet50(pretrained=True)

    #adjust the final layer to output 10 classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)


    #create dataset and dataloaders
    dataset = ImageFolder(root="data/train", transform=transforms.Compose([
        transforms.CenterCrop(512),
        transforms.ToTensor()
    ]))

    #split in train and validation so that we have 10% of every class in the validation set
    train_dataset, cal_dataset = split_dataset(dataset, split_ratio=0.9)
    print("Train dataset size: ", len(train_dataset))
    print("Validation dataset size: ", len(cal_dataset))

    #create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=32, shuffle=False)

    print(dataset[0][1])
    print(dataset[10][1])
    print(dataset[7000][1])
    print(dataset.class_to_idx)

    train(model, device=device, train_loader=train_loader, val_loader=val_loader, epochs=10, batch_size=32)


def train(model, device, train_loader, val_loader, epochs=10, batch_size=32):
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(epochs):
        model.train()
        acc_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            acc_loss += loss.item()
            print("Loss: ", loss.item())
        
        # save the model if the loss is the best so far
        val_loss = evaluate(model, device, val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {acc_loss:.4f}, Validation Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved with validation loss: ", best_val_loss)
        torch.save(model.state_dict(), "last_model.pth")


def evaluate(model, device, val_loader):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)


"""class trainDataset(Dataset):
    def __init__(self, data_dir, transform=None, split_ratio=0.9, train=True):
        self.img_labels = []
        self.imgs = []
        self.transform = transform

        # Iterate through the subdirectories in the data_dir, folder names are the labels
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if os.path.isdir(label_dir):
                for img_file in os.listdir(label_dir):
                    self.img_labels.append(label)
                    self.imgs.append(os.path.join(label_dir, img_file))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        path = self.imgs[idx]
        #image = decode_image(path)
        image = Image.open(path).convert("RGB")
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
""" 
def split_dataset(dataset, split_ratio=0.9):
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size

    return random_split(dataset, [train_size, val_size])



if __name__ == "__main__":
    main()