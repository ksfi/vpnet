import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class VanishingPointNet(nn.Module):
    def __init__(self):
        super(VanishingPointNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=8, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=8, stride=4, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=0)
        self.bn34 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn34(self.conv3(x)))
        x = self.relu(self.bn34(self.conv4(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class VanishingPointDataset(Dataset):
    def __init__(self, optflow_dir, vp_gt, device, xy):
        self.optflow_dir = optflow_dir
        self.optflow_data = dict()
        for idx, file in enumerate(os.listdir(os.fsencode(self.optflow_dir))): self.optflow_data[idx] = os.fsdecode(file)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.data_vp = np.loadtxt(vp_gt)[:,0] if xy == 'x' else np.loadtxt(vp_gt)[:,1]
        self.device = device

    def __len__(self):
        return len(self.data_vp)

    def __getitem__(self, idx):
        flow = cv2.imread(f'{self.optflow_dir}/{self.optflow_data[idx]}')
        flow = self.transform(flow).to(torch.device(self.device))
        vanishing_point = torch.from_numpy(self.data_vp)[idx].to(torch.device(self.device))
        return flow.float(), 0.5*vanishing_point.float()

def main(optflow_dir, vp_gt, model_path, batch_size, learning_rate, epochs, train_ratio, xy):
    model = VanishingPointNet()
    if torch.cuda.is_available():
        device = "cuda"
        model.cuda()
#     if torch.backends.mps.is_available(): trash
#         device = "mps"
    else:
        device = "cpu"

    dataset = VanishingPointDataset(optflow_dir, vp_gt, device, xy)
    train_dataset, test_dataset = train_test_split(dataset, train_size=train_ratio, test_size=1 - train_ratio)
    total_samples = len(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=0.025*np.sqrt((batch_size/len(dataset)*100)))

    train_plot = []
    test_plot = []
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for flows, vanishing_points in tqdm(train_loader):
            if torch.cuda.is_available():
                flows = flows.cuda()
                vanishing_points = vanishing_points.cuda()

            optimizer.zero_grad()
            predictions = model(flows[:,[0,2],:,:])
            loss = criterion(predictions, vanishing_points.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_plot.append(avg_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        model.eval()
        test_loss = 0
        with torch.no_grad():
                for flows, vanishing_points in tqdm(test_loader):
                    if torch.cuda.is_available():
                        device = "cuda"
                        model.cuda()
                    else:
                        device = "cpu"
                    
                    predictions = model(flows[:, [0, 2], :, :])
                    loss = criterion(predictions, vanishing_points.unsqueeze(1))
                    test_loss += loss.item()
            
        avg_test_loss = test_loss / len(test_loader)
        test_plot.append(avg_test_loss)
        print(f'Evaluation - Test Loss: {avg_test_loss:.4f}')
        print("---------")
    x = [_ for _ in range(epochs)]
    plt.plot(x, train_plot)
    plt.savefig(f"train{xy}.png")
    plt.plot(x, test_plot)
    plt.savefig(f"test{xy}.png")
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

def evalmodel(optflow_dir, vp_gt, model_path, batch_size, learning_rate, epochs, train_ratio):
    model = VanishingPointNet()
    model.load_state_dict(torch.load('model.pt'))
    dataset = VanishingPointDataset(optflow_dir, vp_gt)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0
    with torch.no_grad():
        for flows, vanishing_points in tqdm(test_loader):
            if torch.cuda.is_available():
                flows = flows.cuda()
                vanishing_points = vanishing_points.cuda()
            
            predictions = model(flows[:, [0, 2], :, :])
            loss = criterion(predictions, vanishing_points.unsqueeze(1))
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f'Evaluation - Test Loss: {avg_test_loss:.4f}')



if __name__ == '__main__':
    optflow_dir = "_optflow"
    vp_gt = "vp/vps.txt"
    batch_size = 10
    learning_rate = 0.1
    epochs = 100
    train_ratio = 0.8
    for pt in ['x', 'y']:
        model_path = f"model{pt}.pt"
        xy = pt
        main(optflow_dir, vp_gt, model_path, batch_size, learning_rate, epochs, train_ratio, xy)

