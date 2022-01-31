class Queue(list):

    def __init__(self):
        """
        Initialize the list

        Args:
            self: write your description
        """
        self.elements = []

    def add(self, elem):
        """
        Adds an element to the container.

        Args:
            self: write your description
            elem: write your description
        """
        self.elements.append(elem)

    def pop(self):
        """
        Removes the first element from the list and returns it.

        Args:
            self: write your description
        """
        elem = self.elements[0]
        self.elements = self.elements[1:]
        return elem

    def is_empty(self):
        """
        Returns True if the vector has no elements.

        Args:
            self: write your description
        """
        return len(self.elements) == 0
