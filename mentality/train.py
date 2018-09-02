import torch
import torch.utils.data as du
from mentality.observe import TensorBoardObservable
import time
from abc import ABC, abstractmethod

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
            pin_memory=True
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
            if data.shape[0] != batch_size:
                break
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
                if data.shape[0] != batch_size:
                    break
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