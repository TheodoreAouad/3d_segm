from deep_morpho.datasets.mnist_dataset import MnistMorphoDataset
from deep_morpho.morp_operations import ParallelMorpOperations


dataset = MnistMorphoDataset(n_inputs=10000, morp_operation=ParallelMorpOperations.erosion(('disk', 3)))
dataset[0]
