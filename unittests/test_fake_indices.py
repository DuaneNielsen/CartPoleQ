import torch
from unittest import TestCase
import numpy as np
from torch import nn
from mentality import default_maxunpool_indices

def rewrite_index(dims):
    cell = torch.zeros(dims)
    for h in range(dims[0]):
        for w in range(dims[1]):
            cell[h, w] = 2 * 2 * dims[1] * h + 2 * w
    return cell

class TestIndices(TestCase):

    def test_rewrite(self):
        x = np.linspace(0, 15, 16)
        x = x.reshape(4,4)
        cell = rewrite_index((3,3))
        print(cell)
        cell = default_maxunpool_indices((6, 6), (2, 2), 1, 1)
        print(cell)

        mp = nn.MaxPool2d(2,2, return_indices=True)

        a = torch.randn(1, 3, 206,156)
        b = torch.randn(1, 3, 99,74)
        c = torch.randn(1, 3, 33, 16)
        d = torch.randn(1, 3, 8, 12)

        inp = (a,b,c,d)

        for a in inp:
            a, i = mp(a)
            di = default_maxunpool_indices((a.shape[2],a.shape[3]), (2,2), 1, 3)
            print(i.shape, di.shape)
            assert i.shape ==  di.shape



