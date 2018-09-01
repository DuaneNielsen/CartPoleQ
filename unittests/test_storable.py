from unittest import TestCase
from models import AtariConv_v6

class TestStorable(TestCase):
    def test_params(self):
        filter_stack = [128, 128, 64, 64, 64]
        model = AtariConv_v6(filter_stack)
        print(model.config)

    def test_save(self):
        model = AtariConv_v6()
        model.save(str(model.config), 'c:\data')