from unittest import TestCase
import models
import test_model
from mentality import Storeable
import torch

class TestTrainingProgram(TestCase):
    def test_two_models(self):
        fac = test_model.ModelFactoryIterator(models.AtariConv_v6)
        fac.model_args.append(([64, 64, 64, 64, 64],))
        fac.model_args.append(([40, 40, 256, 256, 256],))
        test_model.run(fac, '/spaceinvaders/images/dev/', 2)

    def test_reloading(self):
        model = Storeable.load('C4CP0C45CJ7Z0JHZ', 'c:\data')
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        fac = test_model.OneShotLoader(model, optim)
        test_model.run(fac, '/spaceinvaders/images/dev/', 2)