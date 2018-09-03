import models
import torch
import torchvision
from torchvision import transforms as TVT
from mentality import TensorBoard
from tqdm import tqdm
import os
from urllib.parse import quote


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

    def dataset(self, datapath, ):
        dataset = torchvision.datasets.ImageFolder(
            root=self.DATA_PATH + datapath,
            transform=TVT.Compose([TVT.ToTensor()])
        )
        return dataset



class ModelFactoryIterator:
    def __init__(self, model_type):
        self.model_type = model_type
        self.model_args = []
        self.model_args_index = 0
        self.optimizer_type = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.model_args_index < len(self.model_args):
            model = self.model_type(*self.model_args[self.model_args_index])
            optim =  torch.optim.Adam(model.parameters(), lr=1e-3)
            self.model_args_index += 1
            return model, optim
        else:
            raise StopIteration()

def run(model_factory, dataset_path, epochs):
    jenkins_config = JenkinsConfig()
    print(jenkins_config)
    device = jenkins_config.device()

    dataset = jenkins_config.dataset(dataset_path)

    for model, optim in model_factory:

        run_name = jenkins_config.run_id_string(model)
        model.metadata['run_name'] = run_name
        model.metadata['run_url'] = jenkins_config.run_url_link(model)
        model.metadata['git_commit_hash'] = jenkins_config.GIT_COMMIT
        model.metadata['dataset'] = dataset_path
        tb = TensorBoard(jenkins_config.tb_run_dir(model))
        tb.register(model)

        for epoch in tqdm(range(epochs)):
            model.train_model(dataset, 24, device, optimizer=optim)

            losses = model.test_model(dataset, 24, device)

            l = torch.Tensor(losses)
            model.metadata['ave_test_loss'] = l.mean().item()
            if 'epoch' not in model.metadata:
                model.metadata['epoch'] = 1
            else:
                model.metadata['epoch'] += 1
            model.save(data_dir=jenkins_config.DATA_PATH)


if __name__ == '__main__':
    fac = ModelFactoryIterator(models.AtariConv_v6)
    fac.model_args.append( ([64, 64, 64, 64, 64],) )
    fac.model_args.append( ([40, 40, 256, 256, 256],))

    run(fac, '/spaceinvaders/images/dev/', 2)