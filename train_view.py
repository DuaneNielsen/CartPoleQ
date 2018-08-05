import torchvision
import models
import torch
import monitors


class Dummy(monitors.Controller):
    def forward(self, x):
        self.updateObservers('input', x[0])


class Train(monitors.SummaryWriterWithGlobal):
    def __init__(self):
        monitors.SummaryWriterWithGlobal.__init__(self,'cartpole')
        #dum = Dummy()
        #dum.registerObserver('input',monitors.OpenCV('input'))
        self.input_dims = 3 * 32 * 48
        self.model = self.init_model(self.input_dims)
        self.model.registerObserver('input', monitors.OpenCV('input'))
        self.model.registerObserver('recon', monitors.OpenCV('recon'))
        self.dataset = self.load_dataset()
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=64,
            num_workers=0,
            shuffle=True
        )
        self.loss_function = models.VAE.vae_loss_function
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)


    def init_model(self, input_dims):

        z_dims = 10
        view = models.VAE(input_dims, z_dims)
        return view

    def load_dataset(self):
        data_path = 'cartpole/'
        dataset = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=torchvision.transforms.ToTensor()
        )
        return dataset


    def train(self):
        for batch_idx, (data, target) in enumerate(self.train_loader):
            #train network
            # VAE
            self.tensorboard_step()
            recon_x, mu, logvar = self.model(data)
            loss = self.loss_function(recon_x, data, mu, logvar, self.input_dims, self)
            self.tensorboard_scaler('vae_loss', loss)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def test(self):
        pass

if __name__ == '__main__':
    trainer = Train()
    trainer.train()