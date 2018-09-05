from unittest import TestCase
import models
from mentality import Storeable
from mentality.train import OneShotLoader, ModelFactoryIterator, run
import torch

class TestTrainingProgram(TestCase):
    def test_two_models(self):
        fac = ModelFactoryIterator(models.AtariConv_v6)
        fac.model_args.append(([64, 64, 64, 64, 64],))
        fac.model_args.append(([40, 40, 256, 256, 256],))
        run(fac, 'spaceinvaders/images/dev/', 2)

    def test_reloading(self):
        model = Storeable.load('C4CP0C45CJ7Z0JHZ', 'c:\data')
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        fac = OneShotLoader(model, optim)
        run(fac, 'spaceinvaders/images/dev/', 2)