from mentality.train import ModelFactoryIterator, run
import models

if __name__ == '__main__':
    fac = ModelFactoryIterator(models.AtariConv_v6)
    fac.model_args.append( ([64, 64, 64, 64, 64],) )
    fac.model_args.append( ([40, 40, 256, 256, 256],))

    run(fac, '/spaceinvaders/images/dev/', 5)