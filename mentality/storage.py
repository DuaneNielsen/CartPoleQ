import torch
import os
import pickle
from abc import ABC
import errno
from pathlib import Path
import inspect
import hashlib

class Metadata:
    def __init__(self, obj, args):
        self.args = args
        self.type = type(obj)
        self.data = {}
        self.guid = self.guid(self.type, args)

    """computes a unique GUID for each model/args pair
    """
    def guid(self, obj, args):
        md5 = hashlib.md5()
        md5.update(inspect.getsource(obj).encode('utf8'))
        md5.update(repr(args).encode('utf8'))
        return md5.digest().hex()

    def __str__(self):
        if self.args is not None and len(self.args) > 0:
            string = self.type.__name__  + '/' + str(self.args)
        else:
            string = + str(self.type).__name__
        return string


"""Stores the object params for initialization
Storable MUST be the first in the inheritance chain
So put it as the first class in the inheritance
ie: class MyModel(Storable, nn.Module)
"""
class Storeable(ABC):
    def __init__(self, *args):
        self.classname = type(self)
        self.args = args
        self.metadata = Metadata(self, args)

    """ makes it so we only save the init params and weights to disk
    the res
    """
    def __getstate__(self):
        save_state = []
        save_state.append(self.metadata)
        save_state.append(self.args)
        save_state.append(self.state_dict())
        return save_state

    """ initializes a fresh model from disk with weights
    """
    def __setstate__(self, state):
        self.metadata = state[0]
        self.__init__(*state[1])
        self.load_state_dict(state[2])

    @staticmethod
    def fn(filename, data_dir):
        if data_dir is None:
            home = Path.cwd()
            data = home / "data"
        else:
            data = Path(data_dir)

        if filename is None:
            import random, string
            filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))

        fn = data / "models" / filename
        return fn

    def save(self, filename=None, data_dir=None):
        path = Storeable.fn(filename, data_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f:
            metadata, args, state_dict = self.__getstate__()
            pickle.dump(metadata, f)
            pickle.dump(self, f)
        return path.name

    @staticmethod
    def load(filename, data_dir=None):
        with Storeable.fn(filename, data_dir).open('rb') as f:
            metadata =  pickle.load(f)
            return pickle.load(f)

    """ Load metadata only
    """
    @staticmethod
    def load_metadata(filename, data_dir=None):
        with Storeable.fn(filename, data_dir).open('rb') as f:
            return  pickle.load(f)

"""
ModelConfig is concerned with initializing, loading and saving the model and params
"""

class StoreableOld(ABC):
    def __init__(self, *args):
        self.config = Metadata(type(self), args)

    @staticmethod
    def fn(filename, data_dir=None):
        if data_dir is None:
            data_dir = 'data/models/'

        model_dir = data_dir + '/models/'

        return model_dir + filename + '_config.pt', model_dir + filename + '_model.pt'


    """ Saves the model
    if test_loss is set, it will check that the model on disk has a worse test loss
    before overwriting it
    """
    def save(self, filename, data_dir=None, test_loss=None):

        if data_dir is None:
            data_dir = 'data'

        model_dir = data_dir + '/models/'

        try:
            os.makedirs(model_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        config = self.get_config(filename)
        if config is not None and config.test_loss is not None and config.test_loss < test_loss:
            print('model not saved as the saved model loss was ' + str(config.test_loss) +
                      'which was better than ' + str(test_loss))
            return

        config_filename, model_filename = Storeable.fn(filename, data_dir)

        os.makedirs(os.path.dirname(config_filename), exist_ok=True)

        with open(config_filename, 'wb+') as output:  # Overwrites any existing file.
            pickle.dump(self.config, output, pickle.HIGHEST_PROTOCOL)
        torch.save(self.state_dict(), model_filename)

    def get_config(self, filename):
        config_filename, model_filename = Storeable.fn(filename)
        if Storeable.file_exists(filename):
            with open(config_filename, 'rb') as inp:
                config = pickle.load(inp)
                return config
        else:
            return None


    @staticmethod
    def get_model(config):
        return config.type(*config.params)

    @staticmethod
    def file_exists(filename):
        config_filename, model_filename = Storeable.fn(filename)
        return os.path.isfile(config_filename)

    @staticmethod
    def load(filename, data_dir=None):

        config_filename, model_filename = Storeable.fn(filename, data_dir=data_dir)

        with open(config_filename, 'rb') as input:
            config = pickle.load(input)
            model = Storeable.get_model(config)
            state_dict = torch.load(model_filename)
            model.load_state_dict(state_dict)
        return model

class ModelDb:
    def __init__(self, data_dir):
        self.metadatas = []
        datapath = Path(data_dir) / 'models'
        for file in datapath.iterdir():
            self.metadatas.append(Storeable.load_metadata(file.name, data_dir))

    def print_data(self):
        for metadata in self.metadatas:
            print(metadata)
            print(metadata.guid)
            for field, value in metadata.data.items():
                print(field, value)


