import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset


def main():
    # Setup
    # Identify available device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device: ", device)

    #load pretrained ResNet50 model
    model = models.resnet50(pretrained=True)

    #model = models.resnet50()
    #model.load_state_dict(torch.load("ResNet50.pth"))

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
    
    print(dataset.class_to_idx)

    train(model, device=device, train_loader=train_loader, val_loader=val_loader, epochs=20, batch_size=32)


def train(model, device, train_loader, val_loader, epochs=10, batch_size=32):
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')

    with open("loss.csv", "w") as f:
        f.write("epoch,train_loss,val_loss\n")
    
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

            #Logging the losses to a csv file
            f.write(f"{epoch+1},{acc_loss / len(train_loader)},{val_loss}\n")


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

def split_dataset(dataset, split_ratio=0.9):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                            [0.229,0.224,0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                            [0.229,0.224,0.225])
    ])
    
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size

    # split indices
    train_idx, val_idx = random_split(
        range(len(dataset)),
        [train_size, val_size]
    )

    # create subsets with the respective transforms
    train_dataset = Subset(
        ImageFolder("data/train", transform=train_transform),
        train_idx.indices
    )
    val_dataset = Subset(
        ImageFolder("data/train", transform=val_transform),
        val_idx.indices
    )

    return train_dataset, val_dataset



if __name__ == "__main__":
    main()