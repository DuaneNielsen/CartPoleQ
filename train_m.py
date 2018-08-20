import torch.utils.data as data
import imageio
import torch.nn as nn
from mentality import Observable, OpenCV
import torchvision.transforms.functional as tvf
import models

class VideoFile(data.Dataset):
    def __init__(self, filename):
        self.video = imageio.get_reader(filename,  'ffmpeg')


    def __getitem__(self, index):
        frame = self.video.get_data(index)
        X = tvf.to_tensor(frame)
        return X, X

    def __len__(self):
        return len(self.video)

    def __repr__(self):
        return self.video.get_meta_data()


cartpole1 = VideoFile('data/video/cart/cartpole1.mp4')


def loader(dataset, batch_size):
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True
    )
    return loader

loader = loader(cartpole1, batch_size=2)

print(len(cartpole1))

class Dummy(nn.Module, Observable):
    def __init__(self):
        nn.Module.__init__(self)
        Observable.__init__(self)

    def forward(self, x):
        self.updateObservers('input',x[0],'tensorPIL')
        return x

dum = Dummy()
dum.registerObserver('input', OpenCV('input'))

cvae = models.ConvVAE.load('conv_run2_cart')

for batch, (data, target) in enumerate(loader):
    z = cvae.encode(data)