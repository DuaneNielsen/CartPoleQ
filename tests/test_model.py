from unittest import TestCase
import models
import torch
import torchvision
from torchvision import transforms as TVT
from mentality import TensorBoard
from tqdm import tqdm
import os


class JenkinsConfig:
    def __init__(self):
        self.GIT_COMMIT = os.environ.get('GIT_COMMIT')
        self.DATA_PATH = os.environ.get('DATA_PATH', 'c:\data')
        self.TORCH_DEVICE = os.environ.get('TORCH_DEVICE', 'cuda')

    def run_id_string(self, model):
        return self.GIT_COMMIT + '/' + str(model.config) if self.GIT_COMMIT is not None else str(model.config)

    def __str__(self):
        return 'DATA_PATH ' +  str(self.DATA_PATH) + \
               'GIT_COMMIT ' + str(self.GIT_COMMIT) + \
               'TORCH_DIVICE ' + str(self.TORCH_DEVICE)

def registerViews(model, tb_string):
    tb = TensorBoard(tb_string)
    tb.register(model)


class TestModels(TestCase):

    def test_first_model(self):
        jenkins_config = JenkinsConfig()
        print(jenkins_config)
        device = jenkins_config.TORCH_DEVICE

        spaceinvaders_rgb_210_160 = torchvision.datasets.ImageFolder(
            root= jenkins_config.DATA_PATH + '/spaceinvaders/images/raw/',
            transform=TVT.Compose([TVT.ToTensor()])
        )

        filter_stacks_to_test = []
        filter_stacks_to_test.append([64, 64, 64, 64, 64])
        filter_stacks_to_test.append([16, 32, 64, 64, 64])
        filter_stacks_to_test.append([64, 64, 64, 16, 32])
        filter_stacks_to_test.append([64, 64, 64, 32, 16])
        filter_stacks_to_test.append([64, 32, 16, 32, 64])

        for filter_stack in filter_stacks_to_test:

            atari_conv = models.AtariConv_v6(filter_stack)
            run_name = jenkins_config.run_id_string(atari_conv)
            registerViews(atari_conv, run_name)
            optimizer = torch.optim.Adam(atari_conv.parameters(), lr=1e-3)

            for epoch in tqdm(range(1)):
                atari_conv.train_model(spaceinvaders_rgb_210_160, 24, device, optimizer=optimizer)

                losses = atari_conv.test_model(spaceinvaders_rgb_210_160, 24, device)
                l = torch.Tensor(losses)
                ave_test_loss = l.mean().item()
                atari_conv.save(run_name, ave_test_loss)


