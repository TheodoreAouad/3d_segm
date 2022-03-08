class A:
    def a(self):
        print('A')

class B:
    def b(self):
        print('B')

class C(A, B):
    def __init__(self) -> None:
        self.a()
        super().__init__()