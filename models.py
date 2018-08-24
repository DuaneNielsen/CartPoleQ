import torch
import torch.nn as nn
import torch.nn.functional as F
from mentality import Observable, Storeable, Lossable

# loss should attached to the model, but set during training,
# making it a pure inheritable thing means code changes required to
# test different loss functions
# an inherited variable + an inherited Factory would be worth trying
class BceKldLoss(Lossable):
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss(self, recon_x, mu, logvar, x):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

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



class VAE(nn.Module, Observable, Storeable, BceKldLoss):
    def __init__(self):
        nn.Module.__init__(self)
        Observable.__init__(self)
        Storeable.__init__(self)
        BceKldLoss.__init__(self)

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
class ThreeLayerLinearVAE(nn.Module, Observable, Storeable, BceKldLoss):
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


class ConvVAE(nn.Module, Observable, Storeable, BcelKldLoss):
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


class BaseVAE(nn.Module, Observable, Storeable):
    def __init__(self, encoder, decoder):
        nn.Module.__init__(self)
        Observable.__init__(self)

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        input_shape = x.shape
        self.updateObservers('input', x[0])
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        # should probably make decoder return same shape as encoder
        decoded = decoded.view(input_shape)
        self.updateObservers('output', decoded[0])
        return decoded, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu


"""
input_shape is a tuple of (height,width)
"""
class ConvVAEFixed(BaseVAE, Storeable, BcelKldLoss):
    def __init__(self, input_shape, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
        self.input_shape = input_shape
        encoder = self.Encoder(input_shape, first_kernel, first_stride, second_kernel, second_stride)
        decoder = self.Decoder(encoder.z_shape, first_kernel, first_stride, second_kernel, second_stride)
        BaseVAE.__init__(self, input_shape, encoder, decoder)
        Storeable.__init__(self, input_shape, first_kernel, first_stride, second_kernel, second_stride)


    class Encoder(nn.Module):
        def __init__(self, input_shape, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
            nn.Module.__init__(self)
            # batchnorm in autoencoding is a thing
            # https://arxiv.org/pdf/1602.02282.pdf

            from mentality import conv_output_shape

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

            from mentality import conv_transpose_output_shape

            # decoder
            self.d_conv1 = nn.ConvTranspose2d(32, 32, kernel_size=z_shape, stride=1)
            self.d_bn1 = nn.BatchNorm2d(32)

            self.d_conv2 = nn.ConvTranspose2d(32, 32, kernel_size=second_kernel, stride=second_stride, output_padding=(1,0))
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
class ConvVAE4Fixed(BaseVAE, Storeable, BcelKldLoss):
    def __init__(self, input_shape, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
        self.input_shape = input_shape
        encoder = self.Encoder(input_shape, first_kernel, first_stride, second_kernel, second_stride)
        decoder = self.Decoder(encoder.z_shape, first_kernel, first_stride, second_kernel, second_stride)
        BaseVAE.__init__(self, input_shape, encoder, decoder)
        Storeable.__init__(self, input_shape, first_kernel, first_stride, second_kernel, second_stride)


    class Encoder(nn.Module):
        def __init__(self, input_shape, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
            nn.Module.__init__(self)
            # batchnorm in autoencoding is a thing
            # https://arxiv.org/pdf/1602.02282.pdf

            from mentality import conv_output_shape

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

            from mentality import conv_transpose_output_shape

            # decoder
            self.d_conv1 = nn.ConvTranspose2d(32, 128, kernel_size=z_shape, stride=1)
            self.d_bn1 = nn.BatchNorm2d(128)

            self.d_conv2 = nn.ConvTranspose2d(128, 128, kernel_size=second_kernel, stride=second_stride, output_padding=(1,0))
            self.d_bn2 = nn.BatchNorm2d(128)

            self.d_conv3 = nn.ConvTranspose2d(128, 32, kernel_size=second_kernel, stride=second_stride, output_padding=(1,0))
            self.d_bn3 = nn.BatchNorm2d(32)

            self.d_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=first_kernel, stride=first_stride, output_padding=1)

        def forward(self, z):
            decoded = F.relu(self.d_bn1(self.d_conv1(z)))
            decoded = F.relu(self.d_bn2(self.d_conv2(decoded)))
            decoded = F.relu(self.d_bn3(self.d_conv3(decoded)))
            decoded = self.d_conv4(decoded)
            return decoded


# (210,160)
class AtariConv(BaseVAE, Storeable, BcelKldLoss):
    def __init__(self, input_shape, z_size):
        self.input_shape = input_shape
        self.input_size = self.input_shape[0] * self.input_shape[1]
        encoder = self.Encoder(self.input_shape, z_size)
        decoder = self.Decoder(self.input_shape, z_size)
        BaseVAE.__init__(self, encoder, decoder)
        Storeable.__init__(self, input_shape, z_size)

    class Encoder(nn.Module):
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

    class Decoder(nn.Module):
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
            decoded = F.relu(self.l1(decoded))
            #decoded = decoded.view(-1, self.input_shape[0], self.input_shape[1])
            return decoded


class SimpleLinear(BaseVAE, BcelKldLoss):
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


class PerceptronVAE(BaseVAE, BceKldLoss):
    def __init__(self, input_shape, middle_size, z_size):
        self.input_shape = input_shape
        self.input_size = self.input_shape[0] * self.input_shape[1]
        encoder = self.Encoder(self.input_size, middle_size, z_size)
        decoder = self.Decoder(self.input_size, middle_size, z_size)
        BaseVAE.__init__(self, encoder, decoder)
        Storeable.__init__(self, input_shape, z_size)

    class Encoder(nn.Module):
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

    class Decoder(nn.Module):
        def __init__(self, input_size, middle_size, z_size):
            nn.Module.__init__(self)
            self.middle = nn.Linear(z_size, middle_size)
            self.decoder = nn.Linear(middle_size, input_size)

        def forward(self, z):
            middle = F.relu(self.middle(z))
            return F.sigmoid(self.decoder(middle))

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
        recon =  self.l(x.view(-1, self.input_dims))
        recon = recon.view(shape)
        self.updateObservers('output', recon[0].detach())
        return recon


if __name__ == '__main__':

    from torch.autograd import gradcheck, Variable
    size = (10, 10)
    simpleLinear = SimpleLinear(size, 2).double()
    test_input = (Variable(torch.randn(3, 1, 10, 10).double(), requires_grad=True),)
    test_z = (Variable(torch.randn(3, 1, 2).double(), requires_grad=True),)
    test = gradcheck(simpleLinear.encoder, test_input, eps=1e-6, atol=1e-4)
    test = gradcheck(simpleLinear.decoder, test_z, eps=1e-6, atol=1e-4)
    print('encoder and decoder passed')
    test = gradcheck(simpleLinear, test_input, eps=1e-6, atol=1e-4)

