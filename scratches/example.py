from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets
from torchvision import transforms as T
from torchvision.utils import save_image
from mentality import Observable, OpenCV, Storeable
from scratches.bayes_loss_predictor import LossPredictor

class VAE(nn.Module, Observable, Storeable):
    def __init__(self):
        nn.Module.__init__(self)
        Observable.__init__(self)
        Storeable.__init__(self)

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
        recon =  F.sigmoid(self.fc4(h3))
        output = recon[0].data.view(-1, self.xshape[2], self.xshape[3])
        #self.updateObservers('output', output, 'tensorGreyscale')
        return recon


    def forward(self, x):
        self.xshape = x.shape
        self.updateObservers('input', x[0],'tensorGreyscale')
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        output = recon[0].data.view(-1, self.xshape[2],self.xshape[3])
        self.updateObservers('output', output,'tensorGreyscale')
        return recon, mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE, KLD


def train(epoch, train_loader):
    model.train()

    train_loss = 0
    KLD_loss = 0
    BCE_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        BCD, KLD = loss_function(recon_batch, data, mu, logvar)
        loss = BCD + KLD
        loss.backward()
        train_loss += loss.item()
        KLD_loss += KLD.item()
        BCE_loss += BCD.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
            model.save(filename)



    print('====> Epoch: {} Average loss: {:.4f} BCE: {:.4f} KLD: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset),
            BCE_loss / len(train_loader.dataset), KLD_loss / len(train_loader.dataset)) )
    lp.update(train_loss / len(train_loader.dataset))
    #print(lp.ratio(), lp.past_average(4))

def test(epoch, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            BCE, KLD = loss_function(recon_batch, data, mu, logvar)
            test_loss = BCE.item() + KLD.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    #torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    def load_dataset():
        data_path = 'fullscreen/'
        dataset = datasets.ImageFolder(
            root=data_path,
            transform=T.Compose([T.Grayscale(),T.Resize((28,28)),T.ToTensor()])
        )
        return dataset

    dataset = load_dataset()
    cart_train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    cart_test_loader = cart_train_loader

    mnist_train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=T.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    mnist_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=T.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    lp = LossPredictor()

    filename = 'cartpole_linear1'

    if Storeable.file_exists(filename):
        model = Storeable.load(filename)
    else:
        model = VAE()
        def weights_init(m):
            if type(m) == torch.nn.Conv2d or type(m) == torch.nn.ConvTranspose2d:
                torch.nn.init.kaiming_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        model.apply(weights_init)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if issubclass(type(model), Observable):
        model.registerObserver('input', OpenCV('input'))
        model.registerObserver('output', OpenCV('output'))

    for epoch in range(1, args.epochs + 1):
        train(epoch, cart_train_loader)
        #test(epoch, cart_test_loader)

        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
