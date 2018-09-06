import torch
import torch.nn as nn
import torch.nn.functional as F
from mentality import Dispatcher, Observable, Lossable, Checkable, Trainable, Storeable

# loss should attached to the model, but set during training,
# making it a pure inheritable thing means code changes required to
# test different loss functions
# an inherited variable + an inherited Factory would be worth trying
class BceKldLoss(Lossable):
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss(self, recon_x, mu, logvar, x):
        BCE = F.binary_cross_entropy(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD


class BcelKldLoss(Lossable):
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss(self, recon_x, mu, logvar, x):
        BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD


class BceLoss(Lossable):
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss(self, recon_x, mu, logvar, x):
        BCE = F.binary_cross_entropy(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE #+ KLD


class MSELoss(Lossable):
    def loss(self, recon_x, mu, logvar, x):
        return F.mse_loss(recon_x, x)


class VAE(Storeable, nn.Module, Observable, BceKldLoss):
    def __init__(self):
        nn.Module.__init__(self)
        Observable.__init__(self)
        BceKldLoss.__init__(self)
        Storeable.__init__(self)

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.xshape = None

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        recon =  torch.sigmoid(self.fc4(h3))
        output = recon[0].data.view(-1, self.xshape[2], self.xshape[3])
        #self.updateObservers('decode', output, 'tensorGreyscale')
        return recon

    def forward(self, x):
        self.xshape = x.shape
        self.updateObservers('input', x[0],'tensorGreyscale')
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        recon = recon.view(-1, self.xshape[1], self.xshape[2], self.xshape[3])
        output = recon[0].data
        self.updateObservers('output', output,'tensorGreyscale')
        return recon, mu, logvar

"""
Autoncodes RBG images at 36 * 48
"""
class ThreeLayerLinearVAE(Storeable, nn.Module, Observable, BceKldLoss):
    def __init__(self, input_dims, z_dims):
        nn.Module.__init__(self)
        Observable.__init__(self)
        Storeable.__init__(self, input_dims, z_dims)

        self.input_dims = input_dims

        self.fc1 = nn.Linear(input_dims, 400)
        self.fc12 = nn.Linear(400,400)
        self.fc21 = nn.Linear(400, z_dims)
        self.fc22 = nn.Linear(400, z_dims)

        self.fc3 = nn.Linear(z_dims, 400)
        self.fc31 = nn.Linear(400,400)
        self.fc4 = nn.Linear(400, input_dims)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h12 = F.relu(self.fc12(h1))
        return self.fc21(h12), self.fc22(h12)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h31 = F.relu(self.fc31(h3))
        return torch.sigmoid(self.fc4(h31))

    def forward(self, x):
        input_shape = x.shape
        self.updateObservers('input',x[0])
        mu, logvar = self.encode(x.view(-1, self.input_dims))
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        recon = recon.view(input_shape)
        self.updateObservers('output',recon.data[0])
        return recon, mu, logvar


class ConvVAE(Storeable, nn.Module, Observable, BcelKldLoss):
    def __init__(self, input_dims, z_dims):
        nn.Module.__init__(self)
        Observable.__init__(self)
        Storeable.__init__(self, input_dims, z_dims)

        # batchnorm in autoeconding is a thing
        # https://arxiv.org/pdf/1602.02282.pdf

        # encoder
        self.e_conv1 = nn.Conv2d(3,32, kernel_size=5, stride=2)
        self.e_bn1 = nn.BatchNorm2d(32)

        self.e_conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.e_bn2 = nn.BatchNorm2d(32)

        self.e_mean = nn.Conv2d(32, 32, kernel_size=(5,9), stride=1)
        self.e_logvar = nn.Conv2d(32, 32, kernel_size=(5, 9), stride=1)


        # decoder
        self.d_conv1 = nn.ConvTranspose2d(32, 32, kernel_size=(5,9), stride=1)
        self.d_bn1 = nn.BatchNorm2d(32)

        self.d_conv2 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, output_padding=1)
        self.d_bn2 = nn.BatchNorm2d(32)

        self.d_conv3 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, output_padding=1)

    def encode(self, x):
        encoded = F.relu(self.e_bn1(self.e_conv1(x)))
        encoded = F.relu(self.e_bn2(self.e_conv2(encoded)))

        mean = self.e_mean(encoded)
        logvar = self.e_logvar(encoded)
        return mean, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, shape=None):
        decoded = F.relu(self.d_bn1(self.d_conv1(z)))
        decoded = F.relu(self.d_bn2(self.d_conv2(decoded)))
        h2 = self.d_conv3(decoded)
        h2 = h2.view(shape)
        self.updateObservers('output',h2.data[0])
        return h2

    def forward(self, x):
        input_shape = x.shape
        self.updateObservers('input',x[0])
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, input_shape)
        return recon, mu, logvar


class BaseVAE(nn.Module, Dispatcher, Observable, Trainable):
    def __init__(self, encoder, decoder):
        nn.Module.__init__(self)
        Dispatcher.__init__(self)

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, noise=True):
        input_shape = x.shape
        indices = None
        self.updateObserversWithImage('input', x[0])
        encoded = self.encoder(x)
        mu = encoded[0]
        logvar = encoded[1]

        if mu.shape[1] == 3 or mu.shape[1] == 1:
            self.updateObserversWithImage('z', mu[0].data)
        self.metadata['z_size'] = mu[0].data.numel()

        if len(encoded) > 2:
            indices = encoded[2]

        z = self.reparameterize(mu, logvar, noise=noise)
        if indices is not None:
            decoded = self.decoder(z, indices)
        else:
            decoded = self.decoder(z)

        # should probably make decoder return same shape as encoder
        decoded = decoded.view(input_shape)
        self.updateObserversWithImage('output', decoded[0].data)
        return decoded, mu, logvar

    def reparameterize(self, mu, logvar, noise=True):
        if self.training and noise:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu


"""
input_shape is a tuple of (height,width)
"""
class ConvVAEFixed(Storeable, BaseVAE, MSELoss):
    def __init__(self, input_shape, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
        self.input_shape = input_shape
        encoder = self.Encoder(input_shape, first_kernel, first_stride, second_kernel, second_stride)
        decoder = self.Decoder(encoder.z_shape, first_kernel, first_stride, second_kernel, second_stride)
        BaseVAE.__init__(self, encoder, decoder)
        Storeable.__init__(self, input_shape, first_kernel, first_stride, second_kernel, second_stride)


    class Encoder(nn.Module):
        def __init__(self, input_shape, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
            nn.Module.__init__(self)
            # batchnorm in autoencoding is a thing
            # https://arxiv.org/pdf/1602.02282.pdf

            from mentality.util import conv_output_shape

            # encoder
            self.e_conv1 = nn.Conv2d(3,32, kernel_size=first_kernel, stride=first_stride)
            self.e_bn1 = nn.BatchNorm2d(32)
            output_shape = conv_output_shape(input_shape, kernel_size=first_kernel, stride=first_stride)

            self.e_conv2 = nn.Conv2d(32, 32, kernel_size=second_kernel, stride=second_stride)
            self.e_bn2 = nn.BatchNorm2d(32)
            self.z_shape = conv_output_shape(output_shape, kernel_size=second_kernel, stride=second_stride)

            self.e_mean = nn.Conv2d(32, 32, kernel_size=self.z_shape, stride=1)
            self.e_logvar = nn.Conv2d(32, 32, kernel_size=self.z_shape, stride=1)

        def forward(self, x):
            encoded = F.relu(self.e_bn1(self.e_conv1(x)))
            encoded = F.relu(self.e_bn2(self.e_conv2(encoded)))
            mean = self.e_mean(encoded)
            logvar = self.e_logvar(encoded)
            return mean, logvar

    class Decoder(nn.Module):
        def __init__(self, z_shape, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
            nn.Module.__init__(self)

            # decoder
            self.d_conv1 = nn.ConvTranspose2d(32, 32, kernel_size=z_shape, stride=1)
            self.d_bn1 = nn.BatchNorm2d(32)

            self.d_conv2 = nn.ConvTranspose2d(32, 32, kernel_size=second_kernel, stride=second_stride, output_padding=(0,1))
            self.d_bn2 = nn.BatchNorm2d(32)

            self.d_conv3 = nn.ConvTranspose2d(32, 3, kernel_size=first_kernel, stride=first_stride, output_padding=1)

        def forward(self, z):
            decoded = F.relu(self.d_bn1(self.d_conv1(z)))
            decoded = F.relu(self.d_bn2(self.d_conv2(decoded)))
            decoded = self.d_conv3(decoded)
            return decoded

"""
input_shape is a tuple of (height,width)
"""
class ConvVAE4Fixed(Storeable, BaseVAE, MSELoss):
    def __init__(self, input_shape, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
        self.input_shape = input_shape
        encoder = self.Encoder(input_shape, first_kernel, first_stride, second_kernel, second_stride)
        decoder = self.Decoder(encoder.z_shape, first_kernel, first_stride, second_kernel, second_stride)
        BaseVAE.__init__(self, encoder, decoder)
        Storeable.__init__(self, input_shape, first_kernel, first_stride, second_kernel, second_stride)


    class Encoder(nn.Module):
        def __init__(self, input_shape, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
            nn.Module.__init__(self)
            # batchnorm in autoencoding is a thing
            # https://arxiv.org/pdf/1602.02282.pdf

            from mentality.util import conv_output_shape

            # encoder
            self.e_conv1 = nn.Conv2d(3,32, kernel_size=first_kernel, stride=first_stride)
            self.e_bn1 = nn.BatchNorm2d(32)
            output_shape = conv_output_shape(input_shape, kernel_size=first_kernel, stride=first_stride)

            self.e_conv2 = nn.Conv2d(32, 128, kernel_size=second_kernel, stride=second_stride)
            self.e_bn2 = nn.BatchNorm2d(128)
            output_shape = conv_output_shape(output_shape, kernel_size=second_kernel, stride=second_stride)

            self.e_conv3 = nn.Conv2d(128, 128, kernel_size=second_kernel, stride=second_stride)
            self.e_bn3 = nn.BatchNorm2d(128)
            self.z_shape = conv_output_shape(output_shape, kernel_size=second_kernel, stride=second_stride)

            self.e_mean = nn.Conv2d(128, 32, kernel_size=self.z_shape, stride=1)
            self.e_logvar = nn.Conv2d(128, 32, kernel_size=self.z_shape, stride=1)

        def forward(self, x):
            encoded = F.relu(self.e_bn1(self.e_conv1(x)))
            encoded = F.relu(self.e_bn2(self.e_conv2(encoded)))
            encoded = F.relu(self.e_bn3(self.e_conv3(encoded)))
            mean = self.e_mean(encoded)
            logvar = self.e_logvar(encoded)
            return mean, logvar

    class Decoder(nn.Module):
        def __init__(self, z_shape, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
            nn.Module.__init__(self)

            # decoder
            self.d_conv1 = nn.ConvTranspose2d(32, 128, kernel_size=z_shape, stride=1)
            self.d_bn1 = nn.BatchNorm2d(128)

            self.d_conv2 = nn.ConvTranspose2d(128, 128, kernel_size=second_kernel, stride=second_stride, output_padding=(1,0))
            self.d_bn2 = nn.BatchNorm2d(128)

            self.d_conv3 = nn.ConvTranspose2d(128, 32, kernel_size=second_kernel, stride=second_stride, output_padding=(0,1))
            self.d_bn3 = nn.BatchNorm2d(32)

            self.d_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=first_kernel, stride=first_stride, output_padding=1)

        def forward(self, z):
            decoded = F.relu(self.d_bn1(self.d_conv1(z)))
            decoded = F.relu(self.d_bn2(self.d_conv2(decoded)))
            decoded = F.relu(self.d_bn3(self.d_conv3(decoded)))
            decoded = self.d_conv4(decoded)
            return decoded


# (210,160)
class AtariLinear(BaseVAE, Storeable, BceKldLoss):
    def __init__(self, input_shape, z_size):
        self.input_shape = input_shape
        self.input_size = self.input_shape[0] * self.input_shape[1]
        encoder = self.Encoder(self.input_shape, z_size)
        decoder = self.Decoder(self.input_shape, z_size)
        BaseVAE.__init__(self, encoder, decoder)
        Storeable.__init__(self, input_shape, z_size)

    class Encoder(nn.Module, Checkable):
        def __init__(self, input_shape, z_size):
            nn.Module.__init__(self)
            self.input_shape = input_shape
            self.input_size = self.input_shape[0] * self.input_shape[1]

            self.l1 = nn.Linear(self.input_size, self.input_size)
            self.bn1 = nn.BatchNorm1d(self.input_size)

            self.l2 = nn.Linear(self.input_size, self.input_size)
            self.bn2 = nn.BatchNorm1d(self.input_size)

            self.mu = nn.Linear(self.input_size, z_size)
            self.logvar = nn.Linear(self.input_size, z_size)

        def forward(self, x):
            encoded = x.view(-1, self.input_size)
            encoded = F.relu(self.bn1(self.l1(encoded)))
            encoded = F.relu(self.bn2(self.l2(encoded)))
            mu = F.relu(self.mu(encoded))
            logvar = F.relu(self.logvar(encoded))
            return mu, logvar

    class Decoder(nn.Module, Checkable):
        def __init__(self, input_shape, z_size):
            nn.Module.__init__(self)
            self.input_shape = input_shape
            self.input_size = self.input_shape[0] * self.input_shape[1]
            self.bn_mu = nn.BatchNorm1d(self.input_size)

            self.l1 = nn.Linear(self.input_size, self.input_size)
            self.bn1 = nn.BatchNorm1d(self.input_size)

            self.l2 = nn.Linear(self.input_size, self.input_size)
            self.bn2 = nn.BatchNorm1d(self.input_size)

            self.mu = nn.Linear(z_size, self.input_size)

        def forward(self, encoded):
            decoded = F.relu(self.bn_mu(self.mu(encoded)))
            decoded = F.relu(self.bn2(self.l2(decoded)))
            decoded = torch.sigmoid(self.l1(decoded))
            #decoded = decoded.view(-1, self.input_shape[0], self.input_shape[1])
            return decoded


class SimpleLinear(BaseVAE, Storeable, BcelKldLoss):
    def __init__(self, input_shape, z_size):
        self.input_shape = input_shape
        self.input_size = self.input_shape[0] * self.input_shape[1]
        encoder = self.Encoder(self.input_size, z_size)
        decoder = self.Decoder(self.input_size, z_size)
        BaseVAE.__init__(self, encoder, decoder)
        Storeable.__init__(self, input_shape, z_size)

    class Encoder(nn.Module):
        def __init__(self, input_size, z_size):
            nn.Module.__init__(self)
            self.input_size = input_size
            self.mu = nn.Linear(input_size, z_size)
            self.logvar = nn.Linear(input_size, z_size)

        def forward(self, x):
            xv = x.view(-1, self.input_size)
            mu = F.relu(self.mu(xv))
            logvar = F.relu(self.logvar(xv))
            return mu, logvar

    class Decoder(nn.Module):
        def __init__(self, input_size, z_size):
            nn.Module.__init__(self)
            self.decoder = nn.Linear(z_size, input_size)

        def forward(self, z):
            return F.relu(self.decoder(z))

class PerceptronVAE(Storeable, BaseVAE, MSELoss):
    def __init__(self, input_shape, middle_size, z_size):
        self.input_shape = input_shape
        self.input_size = self.input_shape[0] * self.input_shape[1]
        encoder = self.Encoder(self.input_size, middle_size, z_size)
        decoder = self.Decoder(self.input_size, middle_size, z_size)
        BaseVAE.__init__(self, encoder, decoder)
        Storeable.__init__(self, input_shape, middle_size, z_size)

    class Encoder(nn.Module, Checkable):
        def __init__(self, input_size, middle_size, z_size):
            nn.Module.__init__(self)
            self.input_size = input_size
            self.middle = nn.Linear(input_size, middle_size)
            self.mu = nn.Linear(middle_size, z_size)
            self.logvar = nn.Linear(middle_size, z_size)

        def forward(self, x):
            xv = x.view(-1, self.input_size)
            middle = F.relu(self.middle(xv))
            mu = F.relu(self.mu(middle))
            logvar = F.relu(self.logvar(middle))
            return mu, logvar

    class Decoder(nn.Module, Checkable):
        def __init__(self, input_size, middle_size, z_size):
            nn.Module.__init__(self)
            self.middle = nn.Linear(z_size, middle_size)
            self.decoder = nn.Linear(middle_size, input_size)

        def forward(self, z):
            middle = F.relu(self.middle(z))
            return F.relu(self.decoder(middle))

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3,16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(96, 2)


    def screenToInput(self, screen):
        return screen.permute(0,3,1,2)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.head(x)


class Dummy(Observable):
    def forward(self, x):
        self.updateObservers('input', x[0])


class Linear(torch.nn.Module, Observable):
    def __init__(self, input_dims):
        torch.nn.Module.__init__(self)
        Observable.__init__(self)
        self.input_dims = input_dims

        self.l = torch.nn.Linear(input_dims, input_dims)
        self.loss_function = F.binary_cross_entropy_with_logits

    def forward(self, x):
        self.updateObservers('input', x[0])
        shape = x.shape
        recon = self.l(x.view(-1, self.input_dims))
        recon = recon.view(shape)
        self.updateObservers('output', recon[0].detach())
        return recon


class AtariConv(BaseVAE, Storeable, BceKldLoss):
    def __init__(self):
        self.input_shape = (210, 160)
        encoder = self.Encoder()
        decoder = self.Decoder()
        BaseVAE.__init__(self, encoder, decoder)
        Storeable.__init__(self)

    class Encoder(nn.Module, Checkable):
        def __init__(self):
            nn.Module.__init__(self)
            self.cn1 = nn.Conv2d(3, 32, kernel_size=5, stride=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.mp1 = nn.MaxPool2d(2, 2, return_indices=True)

            self.cn2 = nn.Conv2d(3, 32, kernel_size=5, stride=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.mp2 = nn.MaxPool2d(2, 2, return_indices=False)

        def forward(self, x):
            indices = []
            mu, ind = self.mp1(F.relu(self.bn1(self.cn1(x))))
            indices.append(ind)
            logvar = self.mp2(F.relu(self.bn2(self.cn2(x))))
            return mu, logvar, indices

    class Decoder(nn.Module, Checkable):
        def __init__(self):
            nn.Module.__init__(self)
            self.ct1 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1)
            self.bn1 = nn.BatchNorm2d(3)
            self.up1 = nn.MaxUnpool2d(2, 2)

        def forward(self, z, indices):
            return F.relu(self.bn1(self.ct1(self.up1(z, indices[0]))))

class AtariConv_v2(Storeable, BaseVAE,  MSELoss, Trainable):
    def __init__(self):
        self.input_shape = (210, 160)
        encoder = self.Encoder()
        decoder = self.Decoder()
        BaseVAE.__init__(self, encoder, decoder)
        Storeable.__init__(self)

    class Encoder(nn.Module, Checkable):
        def __init__(self):
            nn.Module.__init__(self)
            self.cn1 = nn.Conv2d(3, 32, kernel_size=5, stride=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.mp1 = nn.MaxPool2d(2, 2, return_indices=True)

            self.cn2 = nn.Conv2d(32, 32, kernel_size=5, stride=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.mp2 = nn.MaxPool2d(2, 2, return_indices=True)

            self.cn3 = nn.Conv2d(32, 16, kernel_size=5, stride=1)
            self.bn3 = nn.BatchNorm2d(16)
            self.mp3 = nn.MaxPool2d(2, 2, return_indices=True)

            self.cn4 = nn.Conv2d(16, 8, kernel_size=5, stride=1)
            self.bn4 = nn.BatchNorm2d(8)
            self.mp4 = nn.MaxPool2d(2, 2, return_indices=True)

            self.l1 = nn.Linear(432, 256)
            self.bn5 = nn.BatchNorm1d(256)

        def forward(self, x):
            indices = []

            #210, 160, 3 => 206, 156, 32 => 103, 78, 32
            encoded = F.relu(self.bn1(self.cn1(x)))
            encoded, ind = self.mp1(encoded)
            indices.append(ind)

            #103, 78, 32 => 99, 74, 32, => 49, 27, 32
            encoded = F.relu(self.bn2(self.cn2(encoded)))
            encoded, ind = self.mp2(encoded)
            indices.append(ind)

            #49, 27, 32 => 44, 33, 16 => 22, 16, 16
            encoded = F.relu(self.bn3(self.cn3(encoded)))
            encoded, ind = self.mp3(encoded)
            indices.append(ind)

            #22, 16, 16 => 18, 12, 8 => 8, 9, 6
            encoded = F.relu(self.bn4(self.cn4(encoded)))
            encoded, ind = self.mp4(encoded)
            indices.append(ind)

            encoded = F.relu(self.bn5(self.l1(encoded.view(-1, 432))))

            return encoded, None, indices

    class Decoder(nn.Module, Checkable):
        def __init__(self):
            nn.Module.__init__(self)

            self.l1 = nn.Linear(256, 432)
            self.bn5 = nn.BatchNorm1d(432)

            self.ct4 = nn.ConvTranspose2d(8, 16, kernel_size=5, stride=1)
            self.bn4 = nn.BatchNorm2d(16)
            self.up4 = nn.MaxUnpool2d(2, 2)

            self.ct3 = nn.ConvTranspose2d(16, 32, kernel_size=5, stride=1)
            self.bn3 = nn.BatchNorm2d(32)
            self.up3 = nn.MaxUnpool2d(2, 2)

            self.ct2 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.up2 = nn.MaxUnpool2d(2, 2)

            self.ct1 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1)
            self.bn1 = nn.BatchNorm2d(3)
            self.up1 = nn.MaxUnpool2d(2, 2)



        def forward(self, z, indices):

            from mentality.util import default_maxunpool_indices

            batch_size = z.shape[0]
            device = z.device

            decoded = F.relu(self.bn5(self.l1(z)))
            decoded = decoded.view(-1, 8, 9, 6)

            indices[3] = default_maxunpool_indices((18,12), (2,2), batch_size, decoded.shape[1], device)
            decoded = self.up4(decoded, indices[3], output_size=(18, 12))
            decoded = F.relu(self.bn4(self.ct4(decoded)))

            indices[2] = default_maxunpool_indices((45, 33), (2, 2), batch_size, decoded.shape[1], device)
            decoded = self.up3(decoded, indices[2], output_size=(45, 33))
            decoded = F.relu(self.bn3(self.ct3(decoded)))

            indices[1] = default_maxunpool_indices((99, 74), (2, 2), batch_size, decoded.shape[1], device)
            decoded = self.up2(decoded, indices[1], output_size=(99,74))
            decoded = F.relu(self.bn2(self.ct2(decoded)))

            indices[0] = default_maxunpool_indices((206, 156), (2, 2), batch_size, decoded.shape[1], device)
            decoded = self.up1(decoded, indices[0], output_size=(206,156))
            decoded = self.bn1(self.ct1(decoded))
            return F.relu(decoded)
            #return torch.sigmoid(decoded)

class AtariConv_v3(BaseVAE, Storeable, MSELoss, Trainable):
    def __init__(self):
        self.input_shape = (210, 160)
        encoder = self.Encoder()
        decoder = self.Decoder()
        BaseVAE.__init__(self, encoder, decoder)
        Storeable.__init__(self)

    class Encoder(nn.Module, Checkable):
        def __init__(self):
            nn.Module.__init__(self)
            self.cn1 = nn.Conv2d(3, 32, kernel_size=5, stride=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.mp1 = nn.Conv2d(32, 32, kernel_size=2, stride=2, groups=32)

            self.cn2 = nn.Conv2d(32, 32, kernel_size=5, stride=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.mp2 = nn.Conv2d(32, 32, kernel_size=2, stride=2, groups=32)

            self.cn3 = nn.Conv2d(32, 16, kernel_size=5, stride=1)
            self.bn3 = nn.BatchNorm2d(16)
            self.mp3 = nn.Conv2d(16, 16, kernel_size=2, stride=2, groups=16)

            self.cn4 = nn.Conv2d(16, 8, kernel_size=5, stride=1)
            self.bn4 = nn.BatchNorm2d(8)
            self.mp4 = nn.Conv2d(8, 8, kernel_size=2, stride=2, groups=8)

            self.l1 = nn.Linear(432, 256)
            self.bn5 = nn.BatchNorm1d(256)

        def forward(self, x):
            indices = []

            #210, 160, 3 => 206, 156, 32 => 103, 78, 32
            encoded = F.relu(self.bn1(self.cn1(x)))
            encoded = self.mp1(encoded)

            #103, 78, 32 => 99, 74, 32, => 49, 37, 32
            encoded = F.relu(self.bn2(self.cn2(encoded)))
            encoded = self.mp2(encoded)

            #49, 37, 32 => 45, 33, 16 => 22, 16, 16
            encoded = F.relu(self.bn3(self.cn3(encoded)))
            encoded = self.mp3(encoded)

            #22, 16, 16 => 18, 12, 8 => 8, 9, 6
            encoded = F.relu(self.bn4(self.cn4(encoded)))
            encoded = self.mp4(encoded)

            encoded = F.relu(self.bn5(self.l1(encoded.view(-1, 432))))

            return encoded, None

    class Decoder(nn.Module, Checkable):
        def __init__(self):
            nn.Module.__init__(self)

            self.l1 = nn.Linear(256, 432)
            self.bn5 = nn.BatchNorm1d(432)

            self.up4 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)
            self.ct4 = nn.ConvTranspose2d(8, 16, kernel_size=5, stride=1)
            self.bn4 = nn.BatchNorm2d(16)

            self.up3 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, output_padding=1)
            self.ct3 = nn.ConvTranspose2d(16, 32, kernel_size=5, stride=1)
            self.bn3 = nn.BatchNorm2d(32)

            self.up2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, output_padding=(1,0))
            self.ct2 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1)
            self.bn2 = nn.BatchNorm2d(32)

            self.up1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
            self.ct1 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1)
            self.bn1 = nn.BatchNorm2d(3)


        def forward(self, z):

            decoded = F.relu(self.bn5(self.l1(z)))
            decoded = decoded.view(-1, 8, 9, 6)

            # output_size=(18, 12)
            decoded = self.up4(decoded)
            decoded = F.relu(self.bn4(self.ct4(decoded)))

            # output_size=(45, 33)
            decoded = self.up3(decoded)
            decoded = F.relu(self.bn3(self.ct3(decoded)))

            # output_size=(99,74)
            decoded = self.up2(decoded)
            decoded = F.relu(self.bn2(self.ct2(decoded)))

            # output_size=(206,156)
            decoded = self.up1(decoded)
            decoded = self.bn1(self.ct1(decoded))
            return F.relu(decoded)


class AtariConv_v4(Storeable, BaseVAE, MSELoss, Trainable):
    def __init__(self):
        self.input_shape = (210, 160)
        encoder = self.Encoder()
        decoder = self.Decoder()
        BaseVAE.__init__(self, encoder, decoder)
        Storeable.__init__(self)

    class Encoder(nn.Module, Checkable):
        def __init__(self):
            nn.Module.__init__(self)
            self.cn1 = nn.Conv2d(3, 32, kernel_size=5, stride=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.mp1 = nn.Conv2d(32, 32, kernel_size=2, stride=2)

            self.cn2 = nn.Conv2d(32, 32, kernel_size=5, stride=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.mp2 = nn.Conv2d(32, 32, kernel_size=2, stride=2)

            self.cn3 = nn.Conv2d(32, 16, kernel_size=5, stride=1)
            self.bn3 = nn.BatchNorm2d(16)
            self.mp3 = nn.Conv2d(16, 16, kernel_size=2, stride=2)

            self.cn4 = nn.Conv2d(16, 8, kernel_size=5, stride=1)
            self.bn4 = nn.BatchNorm2d(8)
            self.mp4 = nn.Conv2d(8, 8, kernel_size=2, stride=2)

            self.cn5 = nn.Conv2d(8, 8, kernel_size=2, stride=2)
            self.bn5 = nn.BatchNorm2d(8)


            self.l1 = nn.Linear(96, 64)
            self.lbn1 = nn.BatchNorm1d(64)

        def forward(self, x):

            #210, 160, 3 => 206, 156, 32 => 103, 78, 32
            encoded = F.relu(self.bn1(self.cn1(x)))
            encoded = self.mp1(encoded)

            #103, 78, 32 => 99, 74, 32, => 49, 37, 32
            encoded = F.relu(self.bn2(self.cn2(encoded)))
            encoded = self.mp2(encoded)

            #49, 37, 32 => 45, 33, 16 => 22, 16, 16
            encoded = F.relu(self.bn3(self.cn3(encoded)))
            encoded = self.mp3(encoded)

            #22, 16, 16 => 18, 12, 8 => 8, 9, 6
            encoded = F.relu(self.bn4(self.cn4(encoded)))
            encoded = self.mp4(encoded)

            #8, 9, 6 => 8, 4, 3
            encoded = self.cn5(encoded)
            encoded = F.relu(self.bn5(encoded))

            encoded = F.relu(self.lbn1(self.l1(encoded.view(-1, 96))))

            return encoded, None

    class Decoder(nn.Module, Checkable):
        def __init__(self):
            nn.Module.__init__(self)

            self.l1 = nn.Linear(64, 96)
            self.lbn1 = nn.BatchNorm1d(96)

            self.ct5 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2, output_padding=(1,0))
            self.bn5 = nn.BatchNorm2d(8)

            self.up4 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)
            self.ct4 = nn.ConvTranspose2d(8, 16, kernel_size=5, stride=1)
            self.bn4 = nn.BatchNorm2d(16)

            self.up3 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, output_padding=1)
            self.ct3 = nn.ConvTranspose2d(16, 32, kernel_size=5, stride=1)
            self.bn3 = nn.BatchNorm2d(32)

            self.up2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, output_padding=(1,0))
            self.ct2 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1)
            self.bn2 = nn.BatchNorm2d(32)

            self.up1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
            self.ct1 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1)
            self.bn1 = nn.BatchNorm2d(3)


        def forward(self, z):

            decoded = F.relu(self.lbn1(self.l1(z)))
            decoded = decoded.view(-1, 8, 4, 3)

            # output_size (8, 9, 6)
            decoded = F.relu(self.bn5(self.ct5(decoded)))

            # output_size=(18, 12)
            decoded = self.up4(decoded)
            decoded = F.relu(self.bn4(self.ct4(decoded)))

            # output_size=(45, 33)
            decoded = self.up3(decoded)
            decoded = F.relu(self.bn3(self.ct3(decoded)))

            # output_size=(99,74)
            decoded = self.up2(decoded)
            decoded = F.relu(self.bn2(self.ct2(decoded)))

            # output_size=(206,156)
            decoded = self.up1(decoded)
            decoded = self.bn1(self.ct1(decoded))
            return F.relu(decoded)

class FireEncoder(nn.Module, Checkable):
    def __init__(self, channels, padding=0):
        nn.Module.__init__(self)
        self.cn1 = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1)
        self.cn1bn = nn.BatchNorm2d(channels)
        self.ds1 = nn.Conv2d(channels, channels, kernel_size=2, stride=2, padding=padding)
        self.ds1bn = nn.BatchNorm2d(channels)
        self.rn1 = nn.Conv2d(channels, 3, kernel_size=1, stride=1)
        self.rn1bn = nn.BatchNorm2d(3)

    def forward(self, x):
        encoded = F.relu(self.cn1bn(self.cn1(x)))
        encoded = F.relu(self.ds1bn(self.ds1(encoded)))
        encoded = F.relu(self.rn1bn(self.rn1(encoded)))
        return encoded

class FireDecoder(nn.Module, Checkable):
    def __init__(self, channels, padding=0, output_padding=0):
        nn.Module.__init__(self)
        self.en1 = nn.Conv2d(3, channels, kernel_size=1, stride=1)
        self.en1bn = nn.BatchNorm2d(channels)
        self.up1 = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2, padding=padding, output_padding=output_padding)
        self.up1bn = nn.BatchNorm2d(channels)
        self.cn1 = nn.ConvTranspose2d(channels, 3, kernel_size=3, stride=1, padding=1)
        self.cn1bn = nn.BatchNorm2d(3)


    def forward(self, y):
        decoded = F.relu(self.en1bn(self.en1(y)))
        decoded = F.relu(self.up1bn(self.up1(decoded)))
        decoded = F.relu(self.cn1bn(self.cn1(decoded)))
        return decoded

#todo we are good to add another layer
"""Based on SqueezeNet
https://arxiv.org/abs/1602.07360
"""
class AtariConv_v5(Storeable, BaseVAE, MSELoss):
    def __init__(self):
        self.input_shape = (210, 160)
        encoder = self.Encoder()
        decoder = self.Decoder()
        BaseVAE.__init__(self, encoder, decoder)
        Storeable.__init__(self)


    class Encoder(nn.Module, Checkable):
        def __init__(self):
            nn.Module.__init__(self)
            self.fe1 = FireEncoder(64)
            self.fe2 = FireEncoder(64, padding=(1,0))

        def forward(self, x):
            # 210, 150 -> 105 80
            encoded = self.fe1(x)
            # 105 80 -> 52, 40
            encoded = self.fe2(encoded)
            return encoded, None

    class Decoder(nn.Module, Checkable):
        def __init__(self):
            nn.Module.__init__(self)
            self.fd1 = FireDecoder(64)
            self.fd2 = FireDecoder(64, padding=(1,0), output_padding=(1,0))

        def forward(self, z):
            decoded = self.fd2(z)
            decoded = self.fd1(decoded)
            return decoded

"""Based on SqueezeNet
https://arxiv.org/abs/1602.07360
"""
class AtariConv_v6(Storeable, BaseVAE, MSELoss):
    def __init__(self, filter_stack=None):
        self.input_shape = (210, 160)
        if filter_stack is None:
            filter_stack = [64, 64, 64, 64, 64]
        encoder = self.Encoder(filter_stack)
        decoder = self.Decoder(filter_stack)
        BaseVAE.__init__(self, encoder, decoder)
        Storeable.__init__(self, filter_stack)





    class Encoder(nn.Module, Checkable):
        def __init__(self, filter_stack):
            nn.Module.__init__(self)
            self.fe1 = FireEncoder(filter_stack[0])
            self.fe2 = FireEncoder(filter_stack[1], padding=(1,0))
            self.fe3 = FireEncoder(filter_stack[2], padding=(1,0))
            self.fe4 = FireEncoder(filter_stack[3], padding=(1, 0))
            self.fe5 = FireEncoder(filter_stack[4], padding=(0, 0))

        def forward(self, x):
            # 210, 160 -> 105 80
            encoded = self.fe1(x)
            # 105 80 -> 52, 40
            encoded = self.fe2(encoded)
            # 52, 40 -> 26, 20
            encoded = self.fe3(encoded)
            # 13, 10
            encoded = self.fe4(encoded)
            # 8, 5
            encoded = self.fe5(encoded)

            return encoded, None

    class Decoder(nn.Module, Checkable):
        def __init__(self, filter_stack):
            nn.Module.__init__(self)
            self.fd1 = FireDecoder(filter_stack[0])
            self.fd2 = FireDecoder(filter_stack[1], padding=(1,0), output_padding=(1,0))
            self.fd3 = FireDecoder(filter_stack[2], padding=(1,0), output_padding=(1,0))
            self.fd4 = FireDecoder(filter_stack[3], padding=(1,0), output_padding=(1,0))
            self.fd5 = FireDecoder(filter_stack[4], padding=(0, 0), output_padding=(0, 0))

        def forward(self, z):
            # 8, 5 -> 15, 10
            decoded = self.fd5(z)
            # 17, 10 -> 29, 20
            decoded = self.fd4(decoded)
            # 26, 20 -> 52, 40
            decoded = self.fd3(decoded)
            # 52, 40 -> 103, 80
            decoded = self.fd2(decoded)

            decoded = self.fd1(decoded)

            return decoded