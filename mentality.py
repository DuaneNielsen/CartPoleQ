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

"""
ModelConfig is concerned with initializing, loading and saving the model and params
"""

class Storeable(ABC):
    def __init__(self, *args):
        self.params = args
        self.type = type(self)
        self.test_loss = None

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
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        torch.save(self.state_dict(), model_filename)

    def get_model(self): #raise NotImplementedError
        return self.type(*self.params)

    @staticmethod
    def file_exists(filename):
        config_filename, model_filename = Storeable.fn(filename)
        return os.path.isfile(config_filename)

    @staticmethod
    def load(filename):

        config_filename, model_filename = Storeable.fn(filename)

        with open(config_filename, 'rb') as input:
            config = pickle.load(input)
            model = config.get_model()
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
        else:
            raise Exception('conversion ' + self.format + ' to numpyRGB not implemented')

    def getImage(self):
        return self.numpyRGB

class TensorPILWrapper(BaseImageWrapper):
    def __init__(self):
        super(TensorPILWrapper, self).__init__()
        self.tensorPIL = None
        if self.format == 'tensorPIL':
            self.tensorPIL = self.image
        #if self.format == 'numpyRGB':
            #self.tensorPIL =  tensorPILTonumpyRBG(self.image)
        else:
            raise Exception('conversion ' + self.format + ' to tensorPIL not implemented')

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

class ImageObserver():
    def __init__(self):
        pass


class ImageFileWriter(ImageObserver):
    def __init__(self, directory, prefix):
        super(ImageFileWriter, self).__init__()
        self.writer = None
        self.directory = directory
        self.prefix = prefix

    def update(self, screen, in_format=None):
        if in_format is None:
            format = self.guess_format(in_format)
        else:
            format = in_format

        if format == 'tensorPIL':
            self.writer = lambda tensor, filename,  : torchvision.utils.save_image(tensor, filename)
        else:
            raise Exception(format + 'not supported yet')

        number = str(random.randint(1,10000))

        self.writer(screen, self.directory + '/' + self.prefix + number + '.png')


class OpenCV(ImageObserver):
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

class BceKldLoss(Lossable):
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss(self, recon_x, mu, logvar, x):
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

class BcelKldLoss(Lossable):
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss(self, recon_x, mu, logvar, x):
        BCE = F.binary_cross_entropy_with_logits(recon_x, x, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD