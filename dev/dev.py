class A:

    def _a(self=None, path):
        print('_a')
    
    def a(self=None):
        A._a(self)


a = A()

a.a()
A.a()