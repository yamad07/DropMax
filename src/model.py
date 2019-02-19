import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.parameterize = BaseCNN()
        self.classifier = BaseCNN()

    def forward(self, x):
        y = self.classifier(x)
        p = self.parameterize(x)
        tau = 1e-1
        u = torch.randn(p.size())
        z = torch.sigmoid(1 / tau * torch.log(p) - torch.log(1 - p) + torch.log(u) - torch.log(1 - u))
        return torch.log(dropmax(y, z))


class BaseCNN(nn.Module):

    def __init__(self):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = x.view(-1, 12 * 12 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def dropmax(y, z, dim=1):
    e = 1e-20
    return (z + e) * torch.exp(y) / ((z + e) * torch.exp(y)).sum(dim).unsqueeze(1).expand(y.size(0), y.size(1))
