from unittest import TestCase
from models import AtariConv_v6
from mentality import Storeable


class Restoreable(Storeable):
    def __init__(self, one, two):
        self.one = one
        self.two = two
        Storeable.__init__(self, one, two)

    def state_dict(self):
        return None

    def load_state_dict(self, thdict):
        pass

class TestStorable(TestCase):
    def test_params(self):
        filter_stack = [128, 128, 64, 64, 64]
        model = AtariConv_v6(filter_stack)

    def test_save(self):
        model = AtariConv_v6()
        import inspect
        print(inspect.getmro(AtariConv_v6))
        model.save('8834739821')
        model = Storeable.load('8834739821')
        assert model is not None


    def test_restore(self):

        r = Restoreable('one','two')
        r.metadata.data['fisk'] = 'frisky'
        r.save('8834739829')
        r = Storeable.load('8834739829')
        print(r.one, r.two)
        assert r is not None and r.one == 'one'
        m = Storeable.load_metadata('8834739829')
        assert m.data['fisk'] == 'frisky'
        print(m)

    def test_save_to_data_dir(self):
        model = AtariConv_v6()
        model.save('8834739821','c:\data')
        model = Storeable.load('8834739821','c:\data')
        assert model is not None
