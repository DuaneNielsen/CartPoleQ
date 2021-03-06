import torch
import torch.utils.data as du
from mentality import Config, ElasticSearchUpdater, TensorBoard, TensorBoardObservable
import time
from abc import ABC, abstractmethod
from tqdm import tqdm

class Lossable(ABC):
    @abstractmethod
    def loss(self, *args):  raise NotImplementedError


class Checkable():
    def __init__(self):
        pass

    """ Builds a random variable for grad_check.
    shape: a tuple with the shape of the input
    batch=True will create a batch of 2 elements, useful if network has batchnorm layers    
    """
    @staticmethod
    def build_input(shape, batch=False):
        from torch.autograd import Variable
        input_shape = shape
        if batch:
            input_shape = (2, *shape)
        return Variable(torch.randn(input_shape).double(), requires_grad=True)

    """Runs a grad check.
    """
    def grad_check(self, *args):
        from torch.autograd import gradcheck
        gradcheck(self.double(), *args, eps=1e-6, atol=1e-4)

class Trainable(TensorBoardObservable):

    @staticmethod
    def loader(dataset, batch_size):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        return loader

    def train_model(self, dataset, batch_size, device, optimizer=None):
        self.to(device)
        self.train()
        #todo: this optimizer resets each epoch, dont reset each epoch!
        if not optimizer:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        train_set = du.Subset(dataset, range(len(dataset) // 10, len(dataset) -1))
        train_loader = self.loader(train_set, batch_size)

        for batch_idx, (data, target) in enumerate(train_loader):
            start = time.time()
            data = data.to(device)
            optimizer.zero_grad()
            output = self(data, noise=False)
            if type(output) == tuple:
                loss = self.loss(*output, data)
            else:
                loss = self.loss(output, data)
            self.writeTrainingLossToTB(loss/data.shape[0])
            loss.backward()
            optimizer.step()
            self.tb_global_step()
            stop = time.time()
            loop_time = stop - start
            self.writePerformanceToTB(loop_time, data.shape[0])

    def test_model(self, dataset, batch_size, device):
        with torch.no_grad():
            self.eval()
            self.to(device)
            test_set = du.Subset(dataset, range(0,len(dataset)//10))
            test_loader = self.loader(test_set, batch_size)
            losses = []

            for batch_idx, (data, target) in enumerate(test_loader):
                start = time.time()
                data = data.to(device)
                output = self(data)
                if type(output) == tuple:
                    loss = self.loss(*output, data)
                else:
                    loss = self.loss(output, data)

                losses.append(loss.item())
                self.writeTestLossToTB(loss/data.shape[0])
                self.tb_global_step()
                stop = time.time()
                loop_time = stop - start
                self.writePerformanceToTB(loop_time, data.shape[0])

            return losses

class Runner(ABC):
    def __iter__(self):
        raise NotImplementedError
    def __next__(self):
        raise NotImplementedError

    def run(self, dataset, batch_size, epochs=2):
        config = Config()
        device = config.device()


        for model, optim in self:

            run_name = config.run_id_string(model)
            model.metadata['run_name'] = run_name
            model.metadata['run_url'] = config.run_url_link(model)
            model.metadata['git_commit_hash'] = config.GIT_COMMIT
            model.metadata['dataset'] = str(dataset.root)
            tb = TensorBoard(config.tb_run_dir(model))
            tb.register(model)
            if 'tb_global_step' in model.metadata:
                tb.global_step = model.metadata['tb_global_step']

            esup = ElasticSearchUpdater()
            esup.register(model)

            for epoch in tqdm(range(epochs)):
                model.train_model(dataset, batch_size=batch_size, device=device, optimizer=optim)

                losses = model.test_model(dataset, batch_size=batch_size, device=device)

                l = torch.Tensor(losses)

                ave_test_loss = l.mean().item()
                import math
                if not math.isnan(ave_test_loss):
                    model.metadata['ave_test_loss'] = ave_test_loss

                if 'epoch' not in model.metadata:
                    model.metadata['epoch'] = 1
                else:
                    model.metadata['epoch'] += 1
                model.metadata['tb_global_step'] = tb.global_step
                model.save(data_dir=config.DATA_PATH)


class ModelFactoryRunner(Runner):
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

class OneShotRunner(Runner):
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer
        self.burned = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.burned:
            raise StopIteration
        else:
            self.burned = True
            if self.optimizer is None:
                optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            else:
                optim = self.optimizer
            return self.model, optim


