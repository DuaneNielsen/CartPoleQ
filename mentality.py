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

"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
"""
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w

def conv_transpose_output_shape(h_w, kernel_size=1, stride=1, pad=0, output_padding=0):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = (h_w[0] - 1) * stride - (2 * pad) + kernel_size[0] + output_padding
    w = (h_w[1] - 1) * stride - (2 * pad) + kernel_size[1] + output_padding
    return h, w

class StoreConfig():
    def __init__(self, object_type, args):
        self.params = args
        self.type = object_type
        self.test_loss = None

"""
ModelConfig is concerned with initializing, loading and saving the model and params
"""

class Storeable(ABC):

    # needs to be fixed so it's not interruptable in the middle of a save!

    def __init__(self, *args):
        self.config = StoreConfig(type(self), args)

    @staticmethod
    def fn(filename):
        return 'data/models/' + filename + '_config.pt', 'data/models/' + filename + '_model.pt'

    def buffer(self, model, test_loss):
        self.state_dict = model.state_dict()
        self.test_loss = test_loss

    def save(self, filename):

        try:
            os.makedirs('data/models')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        config_filename, model_filename = Storeable.fn(filename)

        with open(config_filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self.config, output, pickle.HIGHEST_PROTOCOL)
        torch.save(self.state_dict(), model_filename)

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


class Observable:
    def __init__(self):
        self.pipelineView = {}

    def registerObserver(self, tag, observer):
        if tag not in self.pipelineView:
            self.pipelineView[tag] = []
        self.pipelineView[tag].append(observer)

        return tag,len(self.pipelineView[tag]) -1

    def unregisterObserver(self, id):
        del self.pipelineView[id[0]][id[1]]

    def updateObservers(self, tag, screen, format=None):
        if tag not in self.pipelineView:
            self.pipelineView[tag] = []
        for observer in self.pipelineView[tag]:
            observer.update(screen, format)

    """
    sends a close event to all observers
    used to close video files or save at the end of rollouts
    """
    def endObserverSession(self):
        for tag in self.pipelineView:
            for observer in self.pipelineView[tag]:
                observer.endSession()




class View():
    def __init__(self):
        pass

    def endSession(self):
        pass

class ImageVideoWriter(View):
    def __init__(self, directory, prefix):
        self.directory = directory
        self.prefix = prefix
        self.number = 0
        self.writer = None


    def update(self, screen, in_format=None):
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

    def update(self, screen, in_format=None):

        frame = NumpyRGBWrapper(screen, in_format).numpyRGB
        Image.fromarray(frame).save(self.directory + '/' + self.prefix + str(self.imagenumber) + '.png')
        self.imagenumber = (self.imagenumber + 1) % self.num_images



class OpenCV(View):
    def __init__(self, title='title', screen_resolution=(640,480)):
        super(OpenCV, self).__init__()
        self.C = None
        self.title = title
        self.screen_resolution = screen_resolution

    def update(self, screen, format=None):

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

    def update(self, screen):

        plt.figure(self.figure)

        image = self.C(screen)

        if self.image is None:
            self.image = plt.imshow(image)
        else:
            self.image.set_array(image)

        plt.pause(0.001)
        #plt.draw()


class TensorBoardScalar(View):
    def __init__(self, summary_writer):
        View.__init__(self)
        self.tb = summary_writer

    def update(self, name, scalar_value):
        self.tb.add_scalar(name, scalar_value, self.tb.global_step)


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
    def loss(self,*args):  raise NotImplementedError

