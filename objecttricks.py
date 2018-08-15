class Store:
    def __init__(self, *args):
        self.params = args
        self.type = type(self)

    def hello(self):
        print(self.params)
        print(self.type)

    def load(self):
        object = self.type(*self.params)
        return object

class Stored(Store):
    def __init__(self, parameter1, parmeter2):
        Store.__init__(self, parameter1, parmeter2)



s = Stored('param1','param2')

s.hello()
y = s.load()
y.hello()