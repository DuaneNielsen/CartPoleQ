from unittest import TestCase
import models
import torch
import torchvision
from torchvision import transforms as TVT
from mentality import TensorBoard, OpenCV
from tqdm import tqdm
import os


def registerViews(model, run_name):
    tb = TensorBoard(run_name)
    tb.register(model)
    model.registerView('input', OpenCV('input', (420, 320)))
    model.registerView('output', OpenCV('output', (420, 320)))


class TestModels(TestCase):
    def setUp(self):
        self.data_path = os.environ.get('DATA_PATH')
        if self.data_path is None:
            self.data_path = 'c:\data'

    def test_first_model(self):
        print(self.data_path)
        spaceinvaders_rgb_210_160 = torchvision.datasets.ImageFolder(
            root= self.data_path + '/spaceinvaders/images/raw/',
            transform=TVT.Compose([TVT.ToTensor()])
        )

        device = torch.device("cuda")
        name = 'atari_v5'
        atari_conv = models.AtariConv_v5()
        registerViews(atari_conv, name)
        optimizer = torch.optim.Adam(atari_conv.parameters(), lr=1e-3)

        for epoch in tqdm(range(50)):
            atari_conv.train_model(spaceinvaders_rgb_210_160, 24, device, optimizer=optimizer)

            losses = atari_conv.test_model(spaceinvaders_rgb_210_160, 8, device)
            l = torch.Tensor(losses)
            ave_test_loss = l.mean().item()
            atari_conv.save(name, ave_test_loss)


