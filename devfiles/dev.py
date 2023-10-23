import torch
from torch import nn
from tqdm import tqdm

from deep_morpho.models import GenericLightningModel
from deep_morpho.datasets import DataModule

device = torch.device("cuda:0")


dataloader = DataModule.select("spalikedataset").get_loader(image_size=(256, 256), n_inputs=10000, batch_size=24, num_workers=24)
model = GenericLightningModel.select("LightningBimonnAxspaFromSegm")(
    model_args={
        "bimonn_channels": [7, 7], "bimonn_kernel_size": [7, 7, 7]
    },
    learning_rate=1e-3,
    loss=nn.BCELoss(),
    optimizer=torch.optim.Adam,
)

model.to(device)

pbar = tqdm(dataloader, total=len(dataloader))
for batch in dataloader:
    img, segm, label = batch
    img = img.to(device)
    segm = segm.to(device)
    otp = model(img, segm)
    pbar.update(1)
