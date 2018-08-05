import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dims, z_dims):
        super(VAE, self).__init__()

        self.input_dims = input_dims

        self.fc1 = nn.Linear(input_dims, 400)
        self.fc12 = nn.Linear(400,400)
        self.fc21 = nn.Linear(400, z_dims)
        self.fc22 = nn.Linear(400, z_dims)

        self.fc3 = nn.Linear(z_dims, 400)
        self.fc31 = nn.Linear(400,400)
        self.fc4 = nn.Linear(400, input_dims)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h12 = F.relu(self.fc12(h1))
        return self.fc21(h12), self.fc22(h12)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h31 = F.relu(self.fc31(h3))
        return torch.sigmoid(self.fc4(h31))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dims))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3,16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(96, 2)


    def screenToInput(self, screen):
        return screen.permute(0,3,1,2)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.head(x)