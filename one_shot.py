from mentality import Config, OneShotRunner
from pathlib import Path
import torchvision
import torchvision.transforms as TVT
import models

if __name__ == '__main__':

    config = Config()
    datadir = Path(config.DATA_PATH) / 'spaceinvaders/images/dev'
    dataset = torchvision.datasets.ImageFolder(
        root=datadir.absolute(),
        transform=TVT.Compose([TVT.ToTensor()])
    )

    model = models.AtariLinear((210, 160), 32)
    one_shot = OneShotRunner(model)
    one_shot.run(dataset, batch_size=1, epochs=2)