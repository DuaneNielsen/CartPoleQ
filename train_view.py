import torchvision
import models
import torch
import mentality
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms as TVT

class Train(mentality.SummaryWriterWithGlobal):
    def __init__(self, model, save_name=None):
        mentality.SummaryWriterWithGlobal.__init__(self, 'cartpole')
        self.model = model

        if issubclass(type(self.model), mentality.Storeable) and save_name:
            if mentality.Storeable.file_exists(save_name):
                self.model = mentality.Storeable.load(save_name)

        if issubclass(type(self.model), mentality.Observable):
            self.model.registerObserver('input', mentality.OpenCV('input'))
            self.model.registerObserver('output', mentality.OpenCV('output'))

        self.model = self.model.cuda()
        self.save_name = save_name

    def loader(self, dataset, batch_size):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True
        )
        return loader

    def train(self, dataset, batch_size, optimizer, epochs=1):
        self.model.train()
        train_loader = self.loader(dataset, batch_size)
        for epoch in tqdm(range(epochs)):
            for batch_idx, (data, target) in enumerate(train_loader):
                self.tensorboard_step()
                data = data.cuda()
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

    def test(self, dataset, batch_size, epochs=10):
        self.model.eval()
        test_loader = self.loader(dataset, batch_size)
        for epoch in tqdm(range(epochs)):
            for batch_idx, (data, target) in enumerate(test_loader):
                self.tensorboard_step()
                data = data.cuda()
                output = self.model(data)
                if type(output) == tuple:
                    loss = self.model.loss(*output, data)
                else:
                    loss = self.model.loss(output, data)
                self.tensorboard_scaler('loss/test_loss', loss/data.shape[0])


if __name__ == '__main__':

    def load_dataset(data_path):
        dataset = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=TVT.Compose([TVT.Resize((28,28)),TVT.Grayscale(1),TVT.ToTensor()])
        )
        return dataset

    mnist = torchvision.datasets.MNIST('../data', train=True, download=True,
                   transform=TVT.Compose([TVT.Resize((32,48)),TVT.Grayscale(3),TVT.ToTensor()]))

    cartpole_fullscreen = load_dataset('fullscreen/')

    # custom weights initialization called on netG and netD
    def weights_init(m):
        if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d or type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)



    INPUT_DIMS = 3 * 32 * 48
    Z_DIMS = 32

    model = models.VAE()
    #linear = Linear(INPUT_DIMS)
    #model = models.ThreeLayerLinearVAE(INPUT_DIMS, Z_DIMS)
    #model = models.ConvVAE(INPUT_DIMS,Z_DIMS)
    model.apply(weights_init)
    adam = torch.optim.Adam(model.parameters(), lr=1e-3)


    trainer = Train(model, save_name='delete_me')
    trainer.train(dataset=cartpole_fullscreen, batch_size=2500, optimizer=adam, epochs=600)
    trainer.test(dataset=cartpole_fullscreen, batch_size=2500, epochs=5)
