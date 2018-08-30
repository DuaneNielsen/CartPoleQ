from unittest import TestCase
from mentality import Checkable
import torch
from models import AtariConv_v2, AtariConv

class TestModels(TestCase):
    def test_gradcheck_atari(self):
        i_size = (2, 3, 40, 40)
        z_size = 2

        # atari = AtariLinear(i_size, 2).encoder.grad_check(i_size, batch=True)
        # atari = AtariLinear(i_size, 2).decoder.grad_check(z_size, batch=True)

        AtariConv_v2().encoder.grad_check((Checkable.build_input(i_size),))

        m = AtariConv_v2()
        mu, logvar, indx = AtariConv().encoder(torch.randn(i_size))
        input_var = Checkable.build_input(tuple(mu.shape))
        AtariConv().decoder.grad_check((input_var, indx))

