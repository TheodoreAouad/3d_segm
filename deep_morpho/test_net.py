import os
from os.path import join
from time import time

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from deep_morpho.binarization.bise_closest_selem import ClosestSelemEnum
from deep_morpho.observables import BinaryModeMetricClassif
from deep_morpho.datasets import MnistClassifDataset
from general.nn.pytorch_lightning_module.obs_lightning_module import NetLightning
from deep_morpho.metrics import accuracy


device = "cuda" if torch.cuda.is_available() else "cpu"

exp_paths = "deep_morpho/results/results_tensorboards/"

tb_path = "Bimonn_exp_76/multi/0/bisel/softplus/classif_mnist/version_21"




logger = TensorBoardLogger(join("deep_morpho/results/test_results", tb_path), default_hp_metric=False)

observables = [BinaryModeMetricClassif(metrics={"accuracy": accuracy},
    freq={"test": 1},
    plot_freq={"test": 300},
)]

model_path = join(exp_paths, tb_path, 'checkpoints', os.listdir(join(exp_paths, tb_path, 'checkpoints'))[0])
model = NetLightning.load_from_checkpoint(
    model_path,
    observables=observables.copy(),
    # model_kwargs={"closest_selem_method": ClosestSelemEnum.MIN_DIST}
)
model.to(device)

t1 = time()
print("Updating binary...")
model.model.binary(update_binaries=True)
print(f"Done in {time() - t1:.0f} s.")

trainer = Trainer(
    max_epochs=1,
    gpus=1 if torch.cuda.is_available() else 0,
    logger=logger,
    # progress_bar_refresh_rate=10,
    callbacks=observables.copy(),
    log_every_n_steps=10,
    deterministic=True,
    # num_sanity_val_steps=1,
)

testloader = MnistClassifDataset.get_loader(batch_size=64, train=False, shuffle=False, preprocessing=None, first_idx=0, n_inputs="all", threshold=30, )

trainer.test(model, testloader)
