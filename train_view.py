import torchvision
import models
import torch
import mentality
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms as TVT
import torch.utils.data as du

class Train(mentality.SummaryWriterWithGlobal):
    def __init__(self, model, device, save_name=None):
        run_name = save_name if save_name else 'default'
        mentality.SummaryWriterWithGlobal.__init__(self, run_name)

        self.model = model

        if issubclass(type(model), mentality.Storeable) and save_name:
            if mentality.Storeable.file_exists(save_name):
                self.model = mentality.Storeable.load(save_name)
        else:
            self.model.apply(self.weights_init)


        self.registerObserver('input', mentality.OpenCV('input'))
        self.registerObserver('output', mentality.OpenCV('output'))

        self.model = self.model.to(device)
        self.save_name = save_name

        self.device = device


    def registerObserver(self, tag, view):
        if issubclass(type(self.model), mentality.Observable):
            self.model.registerObserver(tag, view)

    @staticmethod
    # custom weights initialization called on netG and netD
    def weights_init(m):
        if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d or type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)


    def loader(self, dataset, batch_size):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
            pin_memory=True
        )
        return loader

    def train(self, dataset, batch_size):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        train_set = du.Subset(dataset, range(len(dataset) // 5, len(dataset) -1))
        train_loader = self.loader(train_set, batch_size)
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            self.tensorboard_step()
            optimizer.zero_grad()
            output = self.model(data)
            if type(output) == tuple:
                loss = self.model.loss(*output, data)
            else:
                loss = self.model.loss(output, data)
            self.tensorboard_scaler('loss/loss', loss/data.shape[0])
            loss.backward()
            optimizer.step()
            if issubclass(type(self.model), models.Storeable) and self.save_name:
                self.model.save(self.save_name)

    def test(self, dataset, batch_size):
        self.model.eval()
        test_set = du.Subset(dataset, range(0,len(dataset)//5))
        test_loader = self.loader(test_set, batch_size)
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(self.device)
            self.tensorboard_step()
            output = self.model(data)
            if type(output) == tuple:
                loss = self.model.loss(*output, data)
            else:
                loss = self.model.loss(output, data)
            self.tensorboard_scaler('loss/test_loss', loss/data.shape[0])

    def train_test(self, dataset, batch_size, epochs):
        for _ in tqdm(range(epochs)):
            self.train(dataset, batch_size)
            self.test(dataset, batch_size)

    def retest(self, dataset, batch_size, epochs):
        for _ in tqdm(range(epochs)):
            self.test(dataset, batch_size)


if __name__ == '__main__':




    cartpole_rgb_400_600 = torchvision.datasets.ImageFolder(
            root='data/images/',
            transform=TVT.Compose([TVT.ToTensor()])
        )

    cartpole_rgb_100_150 = torchvision.datasets.ImageFolder(
            root='data/images/',
            transform=TVT.Compose([TVT.Resize((100,150)),TVT.ToTensor()])
        )


    cartpole_grescale_28_28 = torchvision.datasets.ImageFolder(
            root='fullscreen/',
            transform=TVT.Compose([TVT.Resize((28,28)),TVT.Grayscale(1),TVT.ToTensor()])
        )

    cartpole_rgb_32_48 = torchvision.datasets.ImageFolder(
            root='fullscreen/',
            transform=TVT.Compose([TVT.ToTensor()])
        )


    mnist_rgb_32_48 = torchvision.datasets.MNIST('../data', train=True, download=True,
                   transform=TVT.Compose([TVT.Resize((32,48)),TVT.Grayscale(3),TVT.ToTensor()]))


    INPUT_DIMS = 3 * 32 * 48
    Z_DIMS = 32
    device = torch.device("cuda")

    #vae = models.VAE()
    #three_linear = models.ThreeLayerLinearVAE(INPUT_DIMS, Z_DIMS)
    # trainer = Train(three_linear, device, save_name='3linear_run1_mnist')

    #conv = models.ConvVAEFixed((400,600))
    #trainer = Train(conv, device, save_name='conv_run4_cart')
    #trainer.train_test(dataset=cartpole_rgb_400_600, batch_size=1, epochs=600)
    #trainer.retest(dataset=cartpole_rgb_400_600, batch_size=16, epochs=10)

    conv_100_150 = models.ConvVAEFixed((100, 150))
    trainer = Train(conv_100_150, device, save_name='conv_100_150_run1_cart')
    trainer.train_test(dataset=cartpole_rgb_100_150, batch_size=8, epochs=600)
