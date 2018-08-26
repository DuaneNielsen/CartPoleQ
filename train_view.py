import torchvision
import models
import torch
from mentality import Storeable, Observable, SummaryWriterWithGlobal, OpenCV, TensorBoardObservable
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms as TVT


#tb = SummaryWriterWithGlobal('default')

#run_name = save_name if save_name else 'default'


# if issubclass(type(model), Storeable) and save_name:
#     if Storeable.file_exists(save_name):
#         self.model = Storeable.load(save_name)
# else:
#     self.model.apply(self.weights_init)
#


# if issubclass(type(self.model), models.Storeable) and self.save_name:
#     self.model.save(self.save_name)


@staticmethod
# custom weights initialization called on netG and netD
def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d or type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


if __name__ == '__main__':

    cartpole_rgb_400_600 = torchvision.datasets.ImageFolder(
        root='data/images/',
        transform=TVT.Compose([TVT.ToTensor()])
    )

    cartpole_rgb_100_150 = torchvision.datasets.ImageFolder(
        root='data/images/cart',
        transform=TVT.Compose([TVT.Resize((100,150)),TVT.ToTensor()])
    )

    cartpole_greycale_28_28 = torchvision.datasets.ImageFolder(
        root='data/images/',
        transform=TVT.Compose([TVT.Resize((28,28)),TVT.Grayscale(1),TVT.ToTensor()])
    )

    cartpole_rgb_32_48 = torchvision.datasets.ImageFolder(
        root='data/images/',
        transform=TVT.Compose([TVT.Resize((32,48)),TVT.ToTensor()])
    )

    spaceinvaders_rgb_210_160 = torchvision.datasets.ImageFolder(
        root='data/images/spaceinvaders',
        transform=TVT.Compose([TVT.ToTensor()])
    )

    spaceinvaders_rgb_100_150 = torchvision.datasets.ImageFolder(
        root='data/images/spaceinvaders',
        transform=TVT.Compose([TVT.Resize((100,150)),TVT.ToTensor()])
    )

    spaceinvaders_rgb_32_48 = torchvision.datasets.ImageFolder(
        root='data/images/spaceinvaders',
        transform=TVT.Compose([TVT.Resize((32,48)),TVT.ToTensor()])
    )

    small = (84, 64)
    spaceinvaders_grey_small = torchvision.datasets.ImageFolder(
        root='data/images/spaceinvaders',
        transform=TVT.Compose([TVT.Grayscale(1), TVT.Resize(small), TVT.ToTensor()])
    )


    mnist_rgb_32_48 = torchvision.datasets.MNIST('../data', train=True, download=True,
                                                 transform=TVT.Compose([TVT.Resize((32,48)),TVT.Grayscale(3),TVT.ToTensor()]))


    mnist_g_32_48 = torchvision.datasets.MNIST('../data', train=True, download=True,
                                               transform=TVT.Compose([TVT.Resize((32, 48)), TVT.ToTensor()]))


    device = torch.device("cuda")

    def registerCV(model):
        model.registerView('input', OpenCV('input'))
        model.registerView('output', OpenCV('output'))


    atari_conv = models.AtariConv_v2()
    registerCV(atari_conv)

    for epoch in tqdm(range(3)):
        atari_conv.train_model(spaceinvaders_rgb_210_160, 56, device)

    #trainer = Train(atari_conv, device, tb, save_name='first_conv_layer_space_invaders_run11')
    #optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.000000001)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 600)
    #optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
    #scheduler = None
    #trainer.retest(dataset=spaceinvaders_rgb_210_160, batch_size=56, epochs=1)

    #three_linear = models.ThreeLayerLinearVAE(INPUT_DIMS, Z_DIMS)
    #trainer = Train(three_linear, device, tb)
    #trainer.train_test(dataset=mnist_rgb_32_48, batch_size=256, epochs=10)

    #conv = models.ConvVAEFixed((400,600))
    #trainer = Train(conv, device, save_name='conv_run4_cart')
    #trainer.train_test(dataset=cartpole_rgb_400_600, batch_size=1, epochs=600)
    #trainer.retest(dataset=cartpole_rgb_400_600, batch_size=16, epochs=10)

    #conv_100_150 = models.ConvVAEFixed((100, 150))
    #trainer = Train(conv_100_150, device, save_name='conv_100_150_run2_cart')
    #trainer.train_test(dataset=cartpole_rgb_100_150, batch_size=256, epochs=600)

    #lin_atari = models.AtariConv(small, 64)
    #trainer = Train(lin_atari, device, save_name='lin_atari_run1_spaceinvaders')
    #trainer.train_test(dataset=spaceinvaders_grey_small, batch_size=2056, epochs=20)

    #lin_atari = models.AtariLinear((32, 48), 32)
    #simple = models.PerceptronVAE((32, 48), 400, 32)
    #trainer = Train(lin_atari, device, tb)
    #trainer.train_test(dataset=mnist_g_32_48, batch_size=2056, epochs=20)


    #three_linear = models.ThreeLayerLinearVAE(INPUT_DIMS, Z_DIMS)
    #trainer = Train(three_linear, device, save_name='3linear_run1_spaceinvaders')
    #trainer.train_test(dataset=spaceinvaders_rgb_32_48, batch_size=1024, epochs=600)

    #conv4_100_150 = models.ConvVAE4Fixed((100, 150))
    #trainer = Train(conv4_100_150, device, save_name='conv_100_150_run3_spaceinv')
    #trainer.train_test(dataset=spaceinvaders_rgb_100_150, batch_size=128, epochs=600)
