from deep_morpho.datasets import MNISTClassical
from torch.utils.data import DataLoader

dataset = MNISTClassical()
# dataloader = DataLoader(dataset, batch_size=32)
dataloader = MNISTClassical.get_loader(batch_size=32)

img, tar = next(iter(dataloader))
pass
