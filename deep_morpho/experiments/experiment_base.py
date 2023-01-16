from abc import ABC, abstractmethod
from typing import Optional, Any, List
import inspect
from os.path import join

from pytorch_lightning import Trainer


class ExperimentBase(ABC):
    def __init__(
        self,
        n_epochs,
        gpus,
        logger,
        observables,
        progress_bar_refresh_rate,
        log_every_n_steps,
        deterministic,
        num_sanity_val_steps,
        model,
        trainloader,
        valloader,
        *args,
        **kwargs
    ):
        self.n_epochs = n_epochs
        self.gpus = gpus
        self.logger = logger
        self.observables = observables
        self.progress_bar_refresh_rate = progress_bar_refresh_rate
        self.log_every_n_steps = log_every_n_steps
        self.deterministic = deterministic
        self.num_sanity_val_steps = num_sanity_val_steps

        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

        self.trainer = None

    @classmethod
    def select_(cls, name: str) -> Optional["ExperimentBase"]:
        """
        Recursive class method iterating over all subclasses to return the
        desired model class.
        """
        if cls.__name__.lower() == name:
            return cls

        for subclass in cls.__subclasses__():
            selected = subclass.select_(name)
            if selected is not None:
                return selected

        return None

    @classmethod
    def select(cls, name: str, **kwargs: Any) -> "ExperimentBase":
        """
        Class method iterating over all subclasses to instantiate the desired
        model.
        """

        selected = cls.select_(name)
        if selected is None:
            raise ValueError("The selected model was not found.")

        return selected(**kwargs)

    @classmethod
    def listing(cls) -> List[str]:
        """List all the available models."""
        subclasses = set()
        if not inspect.isabstract(cls):
            subclasses = {cls.__name__.lower()}

        for subclass in cls.__subclasses__():
            subclasses = subclasses.union(subclass.listing())

        return list(subclasses)

    @abstractmethod
    def setup(self):
        pass

    def launch(self):
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

    def save(self):
        for observable in self.observables:
            observable.save(join(self.trainer.log_dir, 'observables'))

    def run(self):
        self.setup()
        self.launch()
        self.save()
