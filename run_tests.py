from mentality.train import ModelFactoryIterator, OneShotLoader, run, Config
import models

if __name__ == '__main__':

    config = Config()
    config.increment('run_id')

    EPOCHS = 2
    DATASET = 'spaceinvaders\images\dev'


    fac = OneShotLoader(models.AtariConv_v5())
    run(fac, DATASET, EPOCHS)


    fac = OneShotLoader(models.AtariConv_v4())
    run(fac, DATASET, EPOCHS)


    fac = OneShotLoader(models.AtariConv_v3())
    run(fac, DATASET, EPOCHS)


    fac = OneShotLoader(models.AtariConv_v2())
    run(fac, DATASET, EPOCHS)


    fac = OneShotLoader(models.PerceptronVAE((210,160), 32, 32))
    run(fac, DATASET, EPOCHS)


    fac = OneShotLoader(models.ConvVAE4Fixed((210,160)))
    run(fac, DATASET, EPOCHS)


    fac = OneShotLoader(models.ConvVAEFixed((210,160)))
    run(fac, DATASET, EPOCHS)


    fac = ModelFactoryIterator(models.AtariConv_v6)
    fac.model_args.append( ([64, 64, 64, 64, 64],) )
    fac.model_args.append( ([40, 40, 256, 256, 256],))
    run(fac, DATASET, EPOCHS)

