class Queue(list):

    def __init__(self):
        self.elements = []

    def add(self, elem):
        self.elements.append(elem)

    def pop(self):
        elem = self.elements[0]
        self.elements = self.elements[1:]
        return elem

    def is_empty(self):
        return len(self.elements) == 0
