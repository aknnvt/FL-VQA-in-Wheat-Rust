import flwr as fl
from flwr.client import NumPyClient
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from central_model import CentralModel
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class LocalClient(NumPyClient):
    def __init__(self, cid, model, device, data_loader):
        self.cid = cid
        self.model = model
        self.device = device
        self.data_loader = data_loader

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)

    def fit(self, ins):
        self.model.to(self.device)
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001)
        for epoch in range(1):
            for i, data in enumerate(self.data_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        return self.get_parameters(), len(self.data_loader), {}

    def evaluate(self, ins):
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data in self.data_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        accuracy = correct / len(self.data_loader.dataset)
        return {"loss": test_loss / len(self.data_loader), "accuracy": accuracy}

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the dataset directory
dataset_dir = '/path/to/dataset'

# Define the transform for the dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the dataset and data loader
dataset = datasets.ImageFolder(dataset_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create the device (GPU or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the model
model = CentralModel()

# Create the client
client = LocalClient(cid="client1", model=model, device=device, data_loader=data_loader)

# Start the client
fl.client.start_client("0.0.0.0:8080", client=client)