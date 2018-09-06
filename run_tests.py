from mentality.train import ModelFactoryRunner, OneShotRunner, run, Config
import models

if __name__ == '__main__':

    config = Config()
    config.increment('run_id')

    EPOCHS = 2
    DATASET = 'spaceinvaders\images\dev'


    fac = OneShotRunner(models.AtariConv_v5())
    run(fac, DATASET, EPOCHS)


    fac = OneShotRunner(models.AtariConv_v4())
    run(fac, DATASET, EPOCHS)


    fac = OneShotRunner(models.AtariConv_v3())
    run(fac, DATASET, EPOCHS)


    fac = OneShotRunner(models.AtariConv_v2())
    run(fac, DATASET, EPOCHS)


    fac = OneShotRunner(models.PerceptronVAE((210, 160), 32, 32))
    run(fac, DATASET, EPOCHS)


    fac = OneShotRunner(models.ConvVAE4Fixed((210, 160)))
    run(fac, DATASET, EPOCHS)


    fac = OneShotRunner(models.ConvVAEFixed((210, 160)))
    run(fac, DATASET, EPOCHS)


    fac = ModelFactoryRunner(models.AtariConv_v6)
    fac.model_args.append( ([64, 64, 64, 64, 64],) )
    fac.model_args.append( ([40, 40, 256, 256, 256],))
    run(fac, DATASET, EPOCHS)

