import os
from urllib.parse import quote
import torch
import torchvision
from torchvision import transforms as TVT
from pathlib import Path
import logging
from logging.handlers import TimedRotatingFileHandler

class JenkinsConfig:
    def __init__(self):
        # environment variables
        self.BUILD_TAG = os.environ.get('BUILD_TAG', 'build_tag').replace('"', '')
        self.GIT_COMMIT = os.environ.get('GIT_COMMIT', 'git_commit').replace('"', '')
        self.DATA_PATH = os.environ.get('DATA_PATH', 'c:\data').replace('"', '')
        self.TORCH_DEVICE = os.environ.get('TORCH_DEVICE', 'cuda').replace('"', '')

    def run_id_string(self, model):
        return 'runs/' + model.metadata['slug']

    def convert_to_url(self, run, host=None, port='6006'):
        if host is None:
            import socket
            host = socket.gethostname()
        url = run.replace('\\', '\\\\')
        url = run.replace('/', '\\\\')
        url = quote(url)
        url = 'http://' + host + ':' + port + '/#scalars&regexInput=' + url
        return url

    def run_url_link(self, model):
        run = self.run_id_string(model)
        url = self.convert_to_url(run)
        return url

    def tb_run_dir(self, model):
        return self.DATA_PATH + '/' + self.run_id_string(model)

    def device(self):
        return torch.device(str(self.TORCH_DEVICE))

    def __str__(self):
        return 'DATA_PATH ' +  str(self.DATA_PATH) + \
               ' GIT_COMMIT ' + str(self.GIT_COMMIT) + \
               ' TORCH_DEVICE ' + str(self.TORCH_DEVICE)

    def dataset(self, datapath):
        datadir = Path(self.DATA_PATH).joinpath(datapath)
        dataset = torchvision.datasets.ImageFolder(
            root=datadir.absolute(),
            transform=TVT.Compose([TVT.ToTensor()])
        )
        return dataset

    def getLogPath(self, name):
        logfile = Path(self.DATA_PATH) / 'logs' / name
        logfile.parent.mkdir(parents=True, exist_ok=True)
        return logfile
