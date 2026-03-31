import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)
torch.save(model.state_dict(), "ResNet50.pth")