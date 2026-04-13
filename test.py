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
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import sys


def main(**kwargs):
    # Setup
    # Identify available device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device: ", device)

    #load ResNet50 model
    model = models.resnet50()

    #adjust the final layer to output 10 classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

    if len(sys.argv) > 1:
        path = sys.argv[1] 
    else:
        print("Please provide the path to the trained model weights as a command line argument.")
        exit(1)
    #load the trained model weights
    model.load_state_dict(torch.load(path))

    #get dataset
    test_data = CSVImageDataset(csv_file="data/Test Dataset Labels.csv", img_dir="data/test", transform=transforms.Compose([
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize([0.485,0.456,0.406],
                            [0.229,0.224,0.225])
    ]))

    #create dataloader
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    test(model, device=device, test_loader=test_loader, batch_size=32)



class CSVImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.label_to_idx = {
            "Amphibia": 0,
            "Animalia": 1, 
            "Arachnida": 2,
            "Aves": 3,
            "Fungi": 4,
            "Insecta": 5,
            "Mammalia": 6,
            "Mollusca": 7,
            "Plantae": 8,
            "Reptilia": 9
        }

        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]   # filename column
        label = self.data.iloc[idx, 1]      # label column

        label = self.label_to_idx[label]
        label = torch.tensor(label, dtype=torch.long)

        img_path = self.img_dir + "/" + img_name + ".jpg"
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

def test(model, device, test_loader, batch_size=32):
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            print("Processing batch...")
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Save predictions to CSV (file name, predicted label, true label)
    results_df = pd.DataFrame({
        "filename": test_loader.dataset.data.iloc[:, 0],
        "predicted_label": all_preds,
        "true_label": all_labels
    })
    save_path = sys.argv[2]  if len(sys.argv) > 2  else "test_predictions.csv"
    results_df.to_csv(save_path, index=False)

if __name__ == "__main__":
    main()