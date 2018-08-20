import torch
import torch.nn as nn
import torch.nn.functional as F
from mentality import Observable, Storeable, Lossable

class BceKldLoss(Lossable):
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss(self, recon_x, mu, logvar, x):
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD


class BcelKldLoss(Lossable):
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss(self, recon_x, mu, logvar, x):
        BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD



class VAE(nn.Module, Observable, Storeable, BceKldLoss):
    def __init__(self):
        nn.Module.__init__(self)
        Observable.__init__(self)
        Storeable.__init__(self)
        BceKldLoss.__init__(self)

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.xshape = None

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        recon =  torch.sigmoid(self.fc4(h3))
        output = recon[0].data.view(-1, self.xshape[2], self.xshape[3])
        #self.updateObservers('decode', output, 'tensorGreyscale')
        return recon

    def forward(self, x):
        self.xshape = x.shape
        self.updateObservers('input', x[0],'tensorGreyscale')
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        recon = recon.view(-1, self.xshape[1], self.xshape[2], self.xshape[3])
        output = recon[0].data
        self.updateObservers('output', output,'tensorGreyscale')
        return recon, mu, logvar
"""
Autoncodes RBG images at 36 * 48
"""
class ThreeLayerLinearVAE(nn.Module, Observable, Storeable, BceKldLoss):
    def __init__(self, input_dims, z_dims):
        nn.Module.__init__(self)
        Observable.__init__(self)
        Storeable.__init__(self, input_dims, z_dims)

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
        input_shape = x.shape
        self.updateObservers('input',x[0])
        mu, logvar = self.encode(x.view(-1, self.input_dims))
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        recon = recon.view(input_shape)
        self.updateObservers('output',recon.data[0])
        return recon, mu, logvar


class ConvVAE(nn.Module, Observable, Storeable, BcelKldLoss):
    def __init__(self, input_dims, z_dims):
        nn.Module.__init__(self)
        Observable.__init__(self)
        Storeable.__init__(self, input_dims, z_dims)

        # batchnorm in autoeconding is a thing
        # https://arxiv.org/pdf/1602.02282.pdf

        # encoder
        self.e_conv1 = nn.Conv2d(3,32, kernel_size=5, stride=2)
        self.e_bn1 = nn.BatchNorm2d(32)

        self.e_conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.e_bn2 = nn.BatchNorm2d(32)

        self.e_mean = nn.Conv2d(32, 32, kernel_size=(5,9), stride=1)
        self.e_logvar = nn.Conv2d(32, 32, kernel_size=(5, 9), stride=1)


        # decoder
        self.d_conv1 = nn.ConvTranspose2d(32, 32, kernel_size=(5,9), stride=1)
        self.d_bn1 = nn.BatchNorm2d(32)

        self.d_conv2 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, output_padding=1)
        self.d_bn2 = nn.BatchNorm2d(32)

        self.d_conv3 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, output_padding=1)

    def encode(self, x):
        encoded = F.relu(self.e_bn1(self.e_conv1(x)))
        encoded = F.relu(self.e_bn2(self.e_conv2(encoded)))

        mean = self.e_mean(encoded)
        logvar = self.e_logvar(encoded)
        return mean, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, shape=None):
        decoded = F.relu(self.d_bn1(self.d_conv1(z)))
        decoded = F.relu(self.d_bn2(self.d_conv2(decoded)))
        h2 = self.d_conv3(decoded)
        h2 = h2.view(shape)
        self.updateObservers('output',h2.data[0])
        return h2

    def forward(self, x):
        input_shape = x.shape
        self.updateObservers('input',x[0])

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, input_shape)
        return recon, mu, logvar

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

class Dummy(Observable):
    def forward(self, x):
        self.updateObservers('input', x[0])

class Linear(torch.nn.Module, Observable):
    def __init__(self, input_dims):
        torch.nn.Module.__init__(self)
        Observable.__init__(self)
        self.input_dims = input_dims

        self.l = torch.nn.Linear(input_dims, input_dims)
        self.loss_function = F.binary_cross_entropy_with_logits

    def forward(self, x):
        self.updateObservers('input', x[0])
        shape = x.shape
        recon =  self.l(x.view(-1, self.input_dims))
        recon = recon.view(shape)
        self.updateObservers('output', recon[0].detach())
        return recon