import torch
import torchvision
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
import os
import pickle
import errno
from abc import abstractmethod, ABC
import torch.nn.functional as F
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVF
from PIL import Image
import imageio
import torch.utils.data as du
import time

"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
"""
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = floor( ((h_w[0] + (2 * pad[0]) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad[1]) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w

def conv_transpose_output_shape(h_w, kernel_size=1, stride=1, pad=0, output_padding=0):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = (h_w[0] - 1) * stride - (2 * pad) + kernel_size[0] + output_padding
    w = (h_w[1] - 1) * stride - (2 * pad) + kernel_size[1] + output_padding
    return h, w

""" Generates a default index map for nn.MaxUnpool2D operation.
output_shape: the shape that was put into the nn.MaxPool2D operation
in terms of nn.MaxUnpool2D this will be the output_shape
pool_size: the kernel size of the MaxPool2D
"""
def default_maxunpool_indices(output_shape, kernel_size, batch_size, channels, device):
    ph = kernel_size[0]
    pw = kernel_size[1]
    h = output_shape[0]
    w = output_shape[1]
    ih = output_shape[0] // 2
    iw = output_shape[1] // 2
    h_v = torch.arange(ih,dtype=torch.int64, device=device) * pw  * ph * iw
    w_v = torch.arange(iw,dtype=torch.int64, device=device) * pw
    h_v = torch.transpose(h_v.unsqueeze(0), 1,0)
    return (h_v + w_v).expand(batch_size, channels, -1, -1)

class StoreConfig():
    def __init__(self, object_type, args):
        self.params = args
        self.type = object_type
        self.test_loss = None

    def __str__(self):
        if self.params is not None and len(self.params) > 0:
            string = self.type.__name__  + '/' + str(self.params)
        else:
            string = + str(self.type).__name__
        return string


"""
ModelConfig is concerned with initializing, loading and saving the model and params
"""

class Storeable(ABC):
    def __init__(self, *args):
        self.config = StoreConfig(type(self), args)

    @staticmethod
    def fn(filename, data_dir=None):
        if data_dir is None:
            data_dir = 'data/models/'

        model_dir = data_dir + '/models/'

        return model_dir + filename + '_config.pt', model_dir + filename + '_model.pt'


    """ Saves the model
    if test_loss is set, it will check that the model on disk has a worse test loss
    before overwriting it
    """
    def save(self, filename, data_dir=None, test_loss=None):

        if data_dir is None:
            data_dir = 'data'

        model_dir = data_dir + '/models/'

        try:
            os.makedirs(model_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        config = self.get_config(filename)
        if config is not None and config.test_loss is not None and config.test_loss < test_loss:
            print('model not saved as the saved model loss was ' + str(config.test_loss) +
                      'which was better than ' + str(test_loss))
            return

        config_filename, model_filename = Storeable.fn(filename, data_dir)

        os.makedirs(os.path.dirname(config_filename), exist_ok=True)

        with open(config_filename, 'wb+') as output:  # Overwrites any existing file.
            pickle.dump(self.config, output, pickle.HIGHEST_PROTOCOL)
        torch.save(self.state_dict(), model_filename)

    def get_config(self, filename):
        config_filename, model_filename = Storeable.fn(filename)
        if Storeable.file_exists(filename):
            with open(config_filename, 'rb') as inp:
                config = pickle.load(inp)
                return config
        else:
            return None


    @staticmethod
    def get_model(config):
        return config.type(*config.params)

    @staticmethod
    def file_exists(filename):
        config_filename, model_filename = Storeable.fn(filename)
        return os.path.isfile(config_filename)

    @staticmethod
    def load(filename):

        config_filename, model_filename = Storeable.fn(filename)

        with open(config_filename, 'rb') as input:
            config = pickle.load(input)
            model = Storeable.get_model(config)
            state_dict = torch.load(model_filename)
            model.load_state_dict(state_dict)
        return model

tensorPILTonumpyRBG = lambda tensor : tensor.squeeze().permute(1, 2, 0).cpu().numpy()
tensorGreyscaleTonumpyRGB = lambda tensor : tensor.expand(3,-1,-1).squeeze().permute(1, 2, 0).cpu().numpy()

class BaseImageWrapper():
    def __init__(self, image, format=None):
        self.image = image
        self.format = format
        if format is None:
            self.format = self.guess_format(image)

    def guess_format(self, image):
            # guess it based on the screen
            if type(image) == torch.Tensor:
                if image.shape[0] == 3:
                    return 'tensorPIL'
                elif image.shape[0] == 1:
                    return 'tensorGreyscale'
            elif type(image) == np.ndarray:
                if image.shape[0] == 3:
                    return 'numpyRGB'
                elif image.shape[0] == 1:
                    return 'numpyGreyscale'
                elif image.shape[2] == 3:
                    return 'numpyRGB3'
            else:
                raise Exception('failed to autodetect format please specify format')

class NumpyRGBWrapper(BaseImageWrapper):
    def __init__(self, image, format=None):
        super(NumpyRGBWrapper, self).__init__(image, format)
        self.numpyRGB = None
        if self.format == 'numpyRGB':
            self.numpyRGB = self.image
        elif self.format == 'tensorPIL':
            self.numpyRGB =  tensorPILTonumpyRBG(self.image)
        elif self.format == 'tensorGreyscale':
            TF = TVT.Compose([TVT.ToPILImage(),TVT.Grayscale(3),TVT.ToTensor()])
            tensor_PIL = TF(image.cpu())
            self.numpyRGB = tensorPILTonumpyRBG(tensor_PIL)
        elif self.format == 'numpyGreyscale':
            self.numpyRGB = np.repeat(image, 3, axis=0)
        elif self.format == 'numpyRGB3':
            self.numpyRGB = np.transpose(image, [2,0,1])
        else:
            raise Exception('conversion ' + self.format + ' to numpyRGB not implemented')

    def getImage(self):
        return self.numpyRGB

class TensorPILWrapper(BaseImageWrapper):
    def __init__(self, image, format=None):
        BaseImageWrapper.__init__(self, image, format)
        self.tensorPIL = None
        if self.format == 'tensorPIL':
            self.tensorPIL = self.image
        elif self.format == 'numpyRGB':
            # I don't think this works..
            self.tensorPIL =  tensorPILTonumpyRBG(self.image)
        elif self.format == 'numpyRGB3':
            frame = image.transpose(2, 0, 1)
            frame = np.flip(frame, axis=0)
            frame = np.copy(frame)
            TF = TVT.Compose([TVT.ToPILImage(),TVT.ToTensor])
            self.tensorPIL = torch.from_numpy(frame)
        else:
            raise Exception('conversion ' + str(self.format) + ' to tensorPIL not implemented')

    def getImage(self):
        return self.tensorPIL


"""Dispatcher allows dipatch to views.
View's register here
To send a message, inherit Observable and use updateObservers
"""


class Dispatcher:
    def __init__(self):
        self.pipelineView = {}

    def registerView(self, tag, observer):
        if tag not in self.pipelineView:
            self.pipelineView[tag] = []
        self.pipelineView[tag].append(observer)

        return tag,len(self.pipelineView[tag]) -1

    def unregisterView(self, id):
        del self.pipelineView[id[0]][id[1]]


""" Observable provides dispatch method.
To use, make sure the object has a Dispatcher 
"""


class Observable:

    def updateObservers(self, tag, data, metadata=None):
        if hasattr(self, 'pipelineView'):
            if tag not in self.pipelineView:
                self.pipelineView[tag] = []
            for observer in self.pipelineView[tag]:
                observer.update(data, metadata)

    """ Sends a close event to all observers.
    used to close video files or save at the end of rollouts
    """
    def endObserverSession(self):
        if hasattr(self, 'pipelineView'):
            for tag in self.pipelineView:
                for observer in self.pipelineView[tag]:
                    observer.endSession()


""" Abstract base class for implementing View.
"""


class View(ABC):
    @abstractmethod
    def update(self, data, metadata):
        raise NotImplementedError

    def endSession(self):
        pass


class ImageVideoWriter(View):
    def __init__(self, directory, prefix):
        self.directory = directory
        self.prefix = prefix
        self.number = 0
        self.writer = None

    def update(self, screen, metadata=None):

        in_format = metadata['format'] if metadata is not None and 'format' in metadata else None

        if not self.writer:
            self.number += 1
            file = self.directory + self.prefix + str(self.number) + '.mp4'
            self.writer = imageio.get_writer(file, macro_block_size=None)

        frame = NumpyRGBWrapper(screen, in_format).numpyRGB
        self.writer.append_data(frame)

    def endSession(self):
        self.writer.close()
        self.writer = None


class ImageFileWriter(View):
    def __init__(self, directory, prefix, num_images=8192):
        super(ImageFileWriter, self).__init__()
        self.writer = None
        self.directory = directory
        self.prefix = prefix
        self.num_images = num_images
        self.imagenumber = 0

    def update(self, screen, metadata=None):

        in_format = metadata['format'] if metadata is not None and 'format' in metadata else None

        frame = NumpyRGBWrapper(screen, in_format).numpyRGB
        Image.fromarray(frame).save(self.directory + '/' + self.prefix + str(self.imagenumber) + '.png')
        self.imagenumber = (self.imagenumber + 1) % self.num_images



class OpenCV(View):
    def __init__(self, title='title', screen_resolution=(640,480)):
        super(OpenCV, self).__init__()
        self.C = None
        self.title = title
        self.screen_resolution = screen_resolution

    def update(self, screen, metadata=None):

        format = metadata['format'] if metadata is not None and 'format' in metadata else None

        frame = NumpyRGBWrapper(screen, format)
        frame = frame.getImage()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, self.screen_resolution)

        # Display the resulting frame
        cv2.imshow(self.title, frame)
        cv2.waitKey(1)


class Plotter():

    def __init__(self, figure):
        self.image = None
        self.figure = figure
        plt.ion()

    def setInput(self, input):
        if input == 'numpyRGB':
            self.C = lambda x : x

    def update(self, screen, metadata=None):

        plt.figure(self.figure)

        image = self.C(screen)

        if self.image is None:
            self.image = plt.imshow(image)
        else:
            self.image.set_array(image)

        plt.pause(0.001)
        #plt.draw()


class TensorBoard(View, SummaryWriter):
    def __init__(self, run=None, comment='default', image_freq=100):
        View.__init__(self)
        SummaryWriter.__init__(self, run, comment)
        self.image_freq = image_freq
        self.global_step = 0
        self.dispatch = {'tb_step': self.step, 'tb_scalar':self.scalar, 'image':self.image}

    def register(self, model):
        model.registerView('tb_step', self)
        model.registerView('tb_training_loss', self)
        model.registerView('tb_test_loss', self)
        model.registerView('input', self)
        model.registerView('output', self)
        model.registerView('z', self)
        model.registerView('tb_train_time', self)
        model.registerView('tb_train_time_per_item', self)


    def update(self, data, metadata):
        func = self.dispatch.get(metadata['func'])
        func(data, metadata)


    def step(self, data, metadata):
        self.global_step += 1

    def scalar(self, value, metadata):
        self.add_scalar(metadata['name'], value, self.global_step)

    def image(self, value, metadata):
        if self.global_step % self.image_freq == 0:
            self.add_image(metadata['name'], value, self.global_step)

""" Convenience methods for dispatch to tensorboard
requires that the object also inherit Observable
"""


# noinspection PyUnresolvedReferences
class TensorBoardObservable:

    def tb_global_step(self):
        self.updateObservers('tb_step', None, {'func': 'tb_step'})

    def writeScalarToTB(self, tag, value, tb_name):
        self.updateObservers(tag, value,
                             {'func': 'tb_scalar',
                              'name': tb_name})

    def writeTrainingLossToTB(self, loss):
        self.writeScalarToTB('tb_training_loss', loss, 'loss/train')

    def writeTestLossToTB(self, loss):
        self.writeScalarToTB('tb_test_loss', loss, 'loss/test')

    def writePerformanceToTB(self, time, batch_size):
        self.writeScalarToTB('tb_train_time', time, 'perf/train_time_per_batch')
        if batch_size != 0:
            self.writeScalarToTB('tb_train_time_per_item', time/batch_size, 'perf/train_time_per_item')

class SummaryWriterWithGlobal(SummaryWriter):
    def __init__(self, comment):
        super(SummaryWriterWithGlobal, self).__init__(comment=comment)
        self.global_step = 0

    def tensorboard_step(self):
        self.global_step += 1

    def tensorboard_scaler(self, name, scalar):
        self.add_scalar(name, scalar, self.global_step)

    """
    Adds a matplotlib plot to tensorboard
    """
    def plotImage(self, plot):
        self.add_image('Image', plot.getPlotAsTensor(), self.global_step)
        plot.close()


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