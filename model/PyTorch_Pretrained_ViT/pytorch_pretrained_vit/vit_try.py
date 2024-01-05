import torch
from model import ViT
import json
from PIL import Image

from torchvision import transforms

model = ViT('B_32', pretrained=True)
img1 = Image.open('933519976853622784.jpg')
tfms = transforms.Compose([transforms.Resize(model.image_size), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])
img = tfms(img1).unsqueeze(0)
i,a = model(img)
print(i)