import subprocess
from pathlib import Path


parent_dir = Path("deep_morpho/saved_args/recompute_projected")

datasets = [
    "diskorect", 'inverted_mnist', "fashionmnist", "mnist", "mnist_gray"
]

operations = [
    "dilation", "erosion", "opening", "closing", "white_tophat", "black_tophat"
]

for dataset in datasets:
    for operation in operations:
        (parent_dir / dataset / operation).mkdir(parents=True, exist_ok=True)
        print(parent_dir / dataset / operation)