# inference
import torch, torch.nn as nn, torch.nn.functional as F, random, numpy as np
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1    = nn.Conv2d(3, 32, 3, 1)
        self.conv2    = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1      = nn.Linear(614656, 128)
        self.fc2      = nn.Linear(128, 9)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

test_batch_size  = 100
saved_model_path = '0602-656377418-Garg.pt'
device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint       = torch.load(saved_model_path, map_location=device)
model            = Net().to(device)
model.load_state_dict(checkpoint)
model.eval()

transform    = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test_dataset = datasets.ImageFolder('test_original/', transform=transform)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size)

tot_loss = 0
correct  = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output       = model(data)
        tot_loss     += torch.nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
        pred         = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct      += pred.eq(target.view_as(pred)).sum().item()

print('Test Loss: {:.6f}, Test Accuracy: {:.2f}%'.format(
    tot_loss/(len(test_loader)), 100.0*correct/(len(test_loader)*test_batch_size)))