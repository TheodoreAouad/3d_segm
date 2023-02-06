from abc import ABC, abstractmethod
from typing import Optional, Any, List
import inspect
from os.path import join

from pytorch_lightning import Trainer

from deep_morpho.datasets import DataModule
from deep_morpho.models.bimonn import BiMoNN
from deep_morpho.utils import Parser


class ExperimentBase(ABC):
    def __init__(
        self,
        args: Parser,
        datamodule_class=DataModule,
        model_class=BiMoNN,
    ):
        self.args = args
        self.datamodule_class = datamodule_class
        self.model_class = model_class

    def select_model(self, model_name: str) -> BiMoNN:
        return self.model_class.select(model_name)
    
    def select_datamodule(self, datamodule_name: str) -> DataModule:
        return self.datamodule_class.select(datamodule_name)

    @abstractmethod
    def setup(self):
        pass

    def train(self):
        self.trainer = Trainer(
            max_epochs=self.n_epochs,
            gpus=self.gpus,
            logger=self.logger,
            progress_bar_refresh_rate=self.observables,
            callbacks=self.progress_bar_refresh_rate,
            log_every_n_steps=self.log_every_n_steps,
            deterministic=self.deterministic,
            num_sanity_val_steps=self.num_sanity_val_steps,
        )

        self.trainer.fit(self.model, self.trainloader, self.valloader)

    def test(self):
        self.trainer.test(self.model, self.testloader)

    def save(self):
        for observable in self.observables:
            observable.save(join(self.trainer.log_dir, 'observables'))

    def run(self):
        self.setup()
        self.launch()
        self.save()
