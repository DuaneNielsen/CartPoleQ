import torch
import torchvision
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter

tensorPILTonumpyRBG = lambda tensor : tensor.squeeze().permute(1, 2, 0).cpu().numpy()

class BaseImageWrapper():
    def __init__(self, image, format=None):
        self.image = image
        self.format = format
        if format is None:
            self.format = self.guess_format(image)

    def guess_format(self, image):
            # guess it based on the screen
            if type(image) == torch.Tensor:
                return 'tensorPIL'
            elif type(image) == np.ndarray:
                return 'numpyRGB'
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


class Controller:
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
    def __init__(self, title, screen_resolution=(640,480)):
        super(OpenCV, self).__init__()
        self.C = None
        self.title = title
        self.screen_resolution = screen_resolution

    def update(self, screen, format=None):

        frame = NumpyRGBWrapper(screen, format).getImage()
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

    def step(self):
        self.global_step += 1

    def scaler(self, name, scalar):
        self.add_scalar(name, scalar, self.global_step)

    """
    Adds a matplotlib plot to tensorboard
    """
    def plotImage(self, plot):
        self.add_image('Image', plot.getPlotAsTensor(), self.global_step)
        plot.close()