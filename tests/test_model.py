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
        # environment variables
        self.BUILD_TAG = os.environ.get('BUILD_TAG', 'build_tag').replace('"', '')
        self.GIT_COMMIT = os.environ.get('GIT_COMMIT', 'git_commit').replace('"', '')
        self.DATA_PATH = os.environ.get('DATA_PATH', 'c:\data').replace('"', '')
        self.TORCH_DEVICE = os.environ.get('TORCH_DEVICE', 'cuda').replace('"', '')

    def run_id_string(self, model):
        return 'runs/' + self.BUILD_TAG + '/' + self.GIT_COMMIT + '/' \
               + model.metadata['slug']

    def device(self):
        return torch.device(str(self.TORCH_DEVICE))

    def __str__(self):
        return 'DATA_PATH ' +  str(self.DATA_PATH) + \
               ' GIT_COMMIT ' + str(self.GIT_COMMIT) + \
               ' TORCH_DIVICE ' + str(self.TORCH_DEVICE)

def registerViews(model, tb_string):
    tb = TensorBoard(tb_string)
    tb.register(model)


class TestModels(TestCase):

    def test_first_model(self):
        jenkins_config = JenkinsConfig()
        print(jenkins_config)
        device = jenkins_config.device()

        spaceinvaders_rgb_210_160 = torchvision.datasets.ImageFolder(
            root= jenkins_config.DATA_PATH + '/spaceinvaders/images/raw/',
            transform=TVT.Compose([TVT.ToTensor()])
        )

        filter_stacks_to_test = []
        filter_stacks_to_test.append([64, 64, 64, 64, 64])
        filter_stacks_to_test.append([32, 64, 256, 256, 256])
        filter_stacks_to_test.append([32, 64, 256, 512, 512])
        filter_stacks_to_test.append([32, 64, 256, 512, 1024])

        for filter_stack in filter_stacks_to_test:

            atari_conv = models.AtariConv_v6(filter_stack)
            run_name = jenkins_config.run_id_string(atari_conv)
            atari_conv.metadata['run_name'] = run_name
            atari_conv.metadata['git_commit_hash'] = jenkins_config.GIT_COMMIT
            registerViews(atari_conv, run_name)
            optimizer = torch.optim.Adam(atari_conv.parameters(), lr=1e-3)

            for epoch in tqdm(range(10)):
                atari_conv.train_model(spaceinvaders_rgb_210_160, 24, device, optimizer=optimizer)

                losses = atari_conv.test_model(spaceinvaders_rgb_210_160, 24, device)
                l = torch.Tensor(losses)
                atari_conv.metadata['ave_test_loss'] = l.mean().item()
                atari_conv.save(data_dir=jenkins_config.DATA_PATH)


