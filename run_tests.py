from mentality import ModelFactoryRunner, OneShotRunner, Config
import models
from pathlib import Path
import torchvision
import torchvision.transforms as TVT

if __name__ == '__main__':

    config = Config()
    config.increment('run_id')

    datadir = Path(config.DATA_PATH) / 'spaceinvaders/images/dev'
    DATASET = torchvision.datasets.ImageFolder(
        root=datadir.absolute(),
        transform=TVT.Compose([TVT.ToTensor()])
    )

    EPOCHS = 2

    fac = OneShotRunner(models.AtariConv_v5())
    fac.run(DATASET, batch_size=24, epochs=EPOCHS)


    fac = OneShotRunner(models.AtariConv_v4())
    fac.run(DATASET, batch_size=24, epochs=EPOCHS)


    fac = OneShotRunner(models.AtariConv_v3())
    fac.run(DATASET, batch_size=24, epochs=EPOCHS)


    fac = OneShotRunner(models.AtariConv_v2())
    fac.run(DATASET, batch_size=24, epochs=EPOCHS)


    fac = OneShotRunner(models.PerceptronVAE((210, 160), 32, 32))
    fac.run(DATASET, batch_size=24, epochs=EPOCHS)


    fac = OneShotRunner(models.ConvVAE4Fixed((210, 160)))
    fac.run(DATASET, batch_size=24, epochs=EPOCHS)


    fac = OneShotRunner(models.ConvVAEFixed((210, 160)))
    fac.run(DATASET, batch_size=24, epochs=EPOCHS)


    fac = ModelFactoryRunner(models.AtariConv_v6)
    fac.model_args.append( ([64, 64, 64, 64, 64],) )
    fac.model_args.append( ([40, 40, 256, 256, 256],))
    fac.run(DATASET, batch_size=24, epochs=EPOCHS)

