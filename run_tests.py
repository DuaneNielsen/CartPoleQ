from mentality.train import ModelFactoryIterator, OneShotLoader, run
import models

if __name__ == '__main__':

    BATCH_SIZE = 2

    fac = OneShotLoader(models.AtariConv_v5())
    run(fac, 'spaceinvaders\images\dev', BATCH_SIZE)


    fac = OneShotLoader(models.AtariConv_v4())
    run(fac, 'spaceinvaders\images\dev', BATCH_SIZE)

    fac = OneShotLoader(models.AtariConv_v3())
    run(fac, 'spaceinvaders\images\dev', BATCH_SIZE)


    fac = OneShotLoader(models.AtariConv_v2())
    run(fac, 'spaceinvaders\images\dev', BATCH_SIZE)


    fac = OneShotLoader(models.PerceptronVAE((210,160), 32, 32))
    run(fac, 'spaceinvaders\images\dev', BATCH_SIZE)

    fac = OneShotLoader(models.ConvVAE4Fixed((210,160)))
    run(fac, 'spaceinvaders\images\dev', BATCH_SIZE)


    fac = OneShotLoader(models.ConvVAEFixed((210,160)))
    run(fac, 'spaceinvaders\images\dev', BATCH_SIZE)

    fac = OneShotLoader(models.AtariConv())
    run(fac, 'spaceinvaders\images\dev', BATCH_SIZE)

    fac = ModelFactoryIterator(models.AtariConv_v6)
    fac.model_args.append( ([64, 64, 64, 64, 64],) )
    fac.model_args.append( ([40, 40, 256, 256, 256],))

    run(fac, 'spaceinvaders\images\dev', BATCH_SIZE)

