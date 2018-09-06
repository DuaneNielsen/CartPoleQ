class A():
    def __init__(self, one, two):
        self.one = one
        self.two = two

params = ((102,23), 'two')

stuff = ('one', params)

a = A(*stuff[1])
print(a.one)
print(a.two)