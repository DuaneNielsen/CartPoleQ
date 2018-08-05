import torchvision
import models
import torch
import monitors

def init_view():
    input_dims = 3 * 32 * 48
    z_dims = 10
    view = models.VAE(input_dims, z_dims)
    return view

def load_dataset():
    data_path = 'cartpole/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader

class Dummy(monitors.Controller):
    def forward(self, x):
        self.updateObservers('input', x[0])


dum = Dummy()
dum.registerObserver('input', monitors.OpenCV('dummy_input'))

for batch_idx, (data, target) in enumerate(load_dataset()):
    #train network
    dum.forward(data)


