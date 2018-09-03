import pickle
from abc import ABC
from pathlib import Path
import inspect
import hashlib
import unicodedata
import re


def slugify(value, allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces to hyphens.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)


"""Stores the object params for initialization
Storable MUST be the first in the inheritance chain
So put it as the first class in the inheritance
ie: class MyModel(Storable, nn.Module)
the init method must also be called as the LAST one in the sequence..
ie: nn.Module.__init__(self)
    Storable.__init(self, arg1, arg2, etc)
fixing to make less fragile is on todo, but not trivial...
"""
class Storeable(ABC):
    def __init__(self, *args):
        self.classname = type(self)

        #snag the args from the child class during initialization
        stack = inspect.stack()
        child_callable = stack[1][0]
        argname, _, _, argvalues = inspect.getargvalues(child_callable)

        self.repr_string = ""
        for key in argname:
            if key != 'self':
                self.repr_string += ' (' + key + '): ' + str(argvalues[key])

        # too lazy to finish the job and figure out how to store them properly
        # so that they can be used to re-initialize a new class
        # so we will have to pass all the args in for now
        self.args = args
        self.metadata = {}
        self.metadata['guid'] = self.guid()
        self.metadata['classname'] = type(self).__name__
        self.metadata['args'] = self.repr_string
        self.metadata['repr'] = repr(self)
        self.metadata['slug'] = slugify(type(self).__name__ + '-' + self.repr_string)


    def extra_repr(self):
        return self.repr_string


    """computes a unique GUID for each model/args pair
    """
    def guid(self):
        md5 = hashlib.md5()
        md5.update(self.repr_string.encode('utf8'))
        return md5.digest().hex()


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
        self.metadata['filename'] = path.name
        from datetime import datetime
        self.metadata['timestamp'] = datetime.now()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f:
            metadata, args, state_dict = self.__getstate__()
            pickle.dump(metadata, f)
            pickle.dump(self, f)
        return path.name

    @staticmethod
    def load(filename, data_dir=None):
        with Storeable.fn(filename, data_dir).open('rb') as f:
            _ =  pickle.load(f)
            return pickle.load(f)

    """ Load metadata only
    """
    @staticmethod
    def load_metadata(filename, data_dir=None):
        with Storeable.fn(filename, data_dir).open('rb') as f:
            return  pickle.load(f)


class ModelDb:
    def __init__(self, data_dir):
        self.metadatas = []
        self.datapath = Path(data_dir) / 'models'
        self.data_dir = data_dir
        for file in self.datapath.iterdir():
            self.metadatas.append(Storeable.load_metadata(file.name, data_dir))

    def print_data(self):
        for metadata in self.metadatas:
            for field, value in metadata.items():
                print(field, value)

    def print_data_for(self, filename):
        metadata = Storeable.load_metadata(filename, self.data_dir)
        for field, value in metadata.items():
            print(field, value)

    """ syncs data in filesystem to elastic
    Dumb sync, just drops the whole index and rewrites it
    """
    def sync_to_elastic(self, host='localhost', port=9200):
        from elasticsearch import Elasticsearch, ElasticsearchException
        es = Elasticsearch([{'host': host, 'port': port}])
        es.indices.delete(index='test-models', ignore=[400, 404])
        for metadata in self.metadatas:
            try:
                res = es.index(index="test-models", doc_type='model', id=metadata['filename'], body=metadata)
                print(res)
            except ElasticsearchException as es1:
                print(es1)












