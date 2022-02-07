from general.nn.viz import Canva, Element, ElementArrow, ElementImage, ElementGrouper

from .morp_operations import ParallelMorpOperations


class MorpOperationsVizualiser:

    def __init__(self, morp_operations: ParallelMorpOperations):
        self.morp_operations = morp_operations
        self.canva = Canva()


    @property
    def model(self):
        return self.morp_operations
