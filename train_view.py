import torchvision
import torch
import models
from mentality import OpenCV, TensorBoard, Storeable
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms as TVT


@staticmethod
# custom weights initialization called on netG and netD
def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d or type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


if __name__ == '__main__':

    import os
    data_path = os.environ.get('DATA_PATH')
    if data_path is None:
        data_path = 'c:\data'

    cartpole_rgb_400_600 = torchvision.datasets.ImageFolder(
        root=data_path + '/cartpole/images/raw',
        transform=TVT.Compose([TVT.ToTensor()])
    )

    cartpole_rgb_100_150 = torchvision.datasets.ImageFolder(
        root=data_path + '/cartpole/images/raw',
        transform=TVT.Compose([TVT.Resize((100,150)),TVT.ToTensor()])
    )

    cartpole_greycale_28_28 = torchvision.datasets.ImageFolder(
        root=data_path + '/cartpole/images/raw',
        transform=TVT.Compose([TVT.Resize((28,28)),TVT.Grayscale(1),TVT.ToTensor()])
    )

    cartpole_rgb_32_48 = torchvision.datasets.ImageFolder(
        root=data_path + '/cartpole/images/raw',
        transform=TVT.Compose([TVT.Resize((32,48)),TVT.ToTensor()])
    )

    spaceinvaders_rgb_210_160 = torchvision.datasets.ImageFolder(
        root=data_path + '/spaceinvaders/images/raw',
        transform=TVT.Compose([TVT.ToTensor()])
    )

    spaceinvaders_rgb_100_150 = torchvision.datasets.ImageFolder(
        root=data_path + '/spaceinvaders/images/raw',
        transform=TVT.Compose([TVT.Resize((100,150)),TVT.ToTensor()])
    )

    spaceinvaders_rgb_32_48 = torchvision.datasets.ImageFolder(
        root=data_path + '/spaceinvaders/images/raw',
        transform=TVT.Compose([TVT.Resize((32,48)),TVT.ToTensor()])
    )

    small = (84, 64)
    spaceinvaders_grey_small = torchvision.datasets.ImageFolder(
        root=data_path + '/spaceinvaders/images/raw',
        transform=TVT.Compose([TVT.Grayscale(1), TVT.Resize(small), TVT.ToTensor()])
    )


    mnist_rgb_32_48 = torchvision.datasets.MNIST('../data', train=True, download=True,
                                                 transform=TVT.Compose([TVT.Resize((32,48)),TVT.Grayscale(3),TVT.ToTensor()]))


    mnist_g_32_48 = torchvision.datasets.MNIST('../data', train=True, download=True,
                                               transform=TVT.Compose([TVT.Resize((32, 48)), TVT.ToTensor()]))


    device = torch.device("cuda")



    def registerViews(model):
        tb = TensorBoard(comment=str(atari_conv.config))
        tb.register(atari_conv)
        model.registerView('input', OpenCV('input',(320,420)))
        model.registerView('output', OpenCV('output',(320,420)))
        model.registerView('z', OpenCV('z',(320,420)))


    atari_conv = models.AtariConv_v6()
    #atari_conv = Storeable.load(name)
    registerViews(atari_conv)

    optimizer = torch.optim.Adam(atari_conv.parameters(), lr=1e-3)

    for epoch in tqdm(range(50)):
        atari_conv.train_model(spaceinvaders_rgb_210_160, 24, device, optimizer=optimizer)

        losses = atari_conv.test_model(spaceinvaders_rgb_210_160, 24, device)
        l = torch.Tensor(losses)
        ave_test_loss = l.mean().item()
        atari_conv.save(str(atari_conv.config), data_path, ave_test_loss)





