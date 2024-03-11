
from importlib import import_module
import pytorch_lightning as pl
import deep_morpho.models.lightning_bimonn as lbimonn
from deep_morpho.datasets import DataModule

models_old = import_module("deep_morpho.results.results_tensorboards.Bimonn_exp_75.multi.0.bisel.softplus.diskorect.black_tophat.disk.version_0.code.deep_morpho.models.lightning_bimonn")

path = 'deep_morpho/results/results_tensorboards/Bimonn_exp_75/multi/0/bisel/softplus/diskorect/black_tophat/disk/version_0/checkpoints/epoch=0-step=2000.ckpt'
# load pytorhc lightning model
model = models_old.LightningBiMoNN.load_from_checkpoint(path)


dataset = "diskorectdataset"

dataloader = DataModule.select(dataset).get_loader()

