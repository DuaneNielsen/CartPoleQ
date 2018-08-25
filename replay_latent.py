import torch
import models
import pickle
from mentality import OpenCV
import os


device = torch.device('cuda')
cvae = models.ConvVAE.load('conv_run2_cart').to(device)
cvae.registerView('output', OpenCV('decode'))

directory = os.fsencode('data/cart/latent/')


for file in os.listdir(directory):
    a, z = pickle.load(open(os.path.join(directory, file),'rb'))
    a = a.to(device)
    z = z.to(device)

    for row, _ in enumerate(z):
        y = cvae.decode(z[row],(1,3,100,150))