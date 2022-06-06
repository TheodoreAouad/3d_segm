""" Script to show a memory leak bug in lightning with BCELoss and logging into tensorboard.
Try running the script with a "wath nvidia-smi", and look at the GPU memory usage increasing.
Try again with the MSELoss, the GPU memory usage does not increase.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from pytorch_lightning.loggers import TensorBoardLogger


from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule


batch_size = 25600


class ARandomNet(LightningModule):

    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.FloatTensor(np.random.random((batch_size, 1))), requires_grad=True)
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.BCELoss()

    def training_step(self, batch, batch_idx):
        label = torch.tensor(np.random.random((batch_size, 1)), requires_grad=True).to("cuda").float()
        # output = torch.tensor(np.random.random((batch_size, 1)), requires_grad=True).to("cuda").float()
        loss = self.loss_fn(label, self.param)
        self.log("loss", loss.item())

        return {'loss': loss}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.01, )


class ARandomDataset(Dataset):

    def __getitem__(self, index):
        return torch.rand(1, 1)
    
    def __len__(self):
        return int(1e6)


dataloader =DataLoader(ARandomDataset(), batch_size=256)
model = ARandomNet()

logger = TensorBoardLogger("todelete", name="test_lightning", default_hp_metric=False)


trainer = Trainer(max_epochs=1000, gpus=1, logger=logger)
trainer.fit(model, dataloader)
