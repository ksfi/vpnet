import torch
import VPNet
import os
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms

model = VPNet.VanishingPointNet()
model_path = './models/modelx.pt' 
w = torch.load(model_path, map_location='cpu')
model.load_state_dict(w)
model.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

if __name__ == "__main__":
    for k in range(0,1):
        directory = os.fsencode(f"_eval/{k}")
        for file in os.listdir(directory):
             filename = os.fsdecode(file)
             if filename.endswith(".png"):
                 print(filename)
                 file = np.array(cv2.imread(f"_eval/{k}/{filename}"))
                 print(file.shape)
                 with torch.no_grad():
                     flow = transform(file).cpu()
                     out = torch.softmax(model.forward(flow), dim=1).cpu().numpy()
                     print(out)
