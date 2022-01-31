import pathlib
from os.path import join
from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl

from general.nn.observables import Observable


class CountInputs(Observable):

    def __init__(self):
        """
        Initialize the state of the class.

        Args:
            self: write your description
        """
        super().__init__()
        self.n_inputs = 0

    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: 'STEP_OUTPUT',
        batch: 'Any',
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """
        Add scalars to the experiment.

        Args:
            self: write your description
            trainer: write your description
            pl_module: write your description
            outputs: write your description
            batch: write your description
            batch_idx: write your description
            dataloader_idx: write your description
        """
        self.n_inputs += len(batch[0])
        trainer.logger.experiment.add_scalar("n_inputs", self.n_inputs, trainer.global_step)

    def save(self, save_path: str):
        """
        Save the model to a file.

        Args:
            self: write your description
            save_path: write your description
        """
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        with open(join(final_dir, str(self.n_inputs)), "w"):
            pass

        return self.n_inputs
