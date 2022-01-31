from typing import Dict, Callable, List

from .bimonn import BiMoNN
from general.nn.pytorch_lightning_module.obs_lightning_module import NetLightning
from ..loss import ThresholdPenalization


class LightningBiMoNN(NetLightning):

    def __init__(
        self,
        model_args: Dict,
        learning_rate: float,
        loss: Callable,
        optimizer: Callable,
        output_dir: str,
        do_thresh_penalization: bool = True,
        args_thresh_penalization: Dict = {
            'coef': .5,
            'degree': 2,
            'detach_weights': True,
        },
        first_batch_pen: int = 100,
        optimizer_args: Dict = {},
        observables: List["Observable"] = [],
    ):
        """
        Initialize the BiMoNN model.

        Args:
            self: write your description
            model_args: write your description
            learning_rate: write your description
            loss: write your description
            optimizer: write your description
            output_dir: write your description
            do_thresh_penalization: write your description
            args_thresh_penalization: write your description
            first_batch_pen: write your description
            optimizer_args: write your description
            observables: write your description
            List: write your description
        """
        super().__init__(
            model=BiMoNN(**model_args),
            learning_rate=learning_rate,
            loss=loss,
            optimizer=optimizer,
            output_dir=output_dir,
            optimizer_args=optimizer_args,
            observables=observables,
        )

        self.do_thresh_penalization = do_thresh_penalization
        self.args_thresh_penalization = args_thresh_penalization
        self.first_batch_pen = first_batch_pen
        if self.do_thresh_penalization:
            self.pen_fn = ThresholdPenalization(
                dilation_layers=self.model.bise, **self.args_thresh_penalization
            )
        self.save_hyperparameters()

        # new_hparams = {}
        # for key, value in self.hparams.items():
        #     if isinstance(value, dict):
        #         for key2, value2 in value.items():
        #             new_hparams[f'{key}/{key2}'] = value2
        # self.hparams.update(new_hparams)

    def obs_training_step(self, batch, batch_idx):
        """
        Perform obs training step.

        Args:
            self: write your description
            batch: write your description
            batch_idx: write your description
        """
        x, y = batch
        # predictions = self.forward(x).squeeze()
        predictions = self.forward(x)

        outputs = {}

        loss_supervised = self.loss(predictions, y)
        outputs['loss_supervised'] = loss_supervised
        outputs['loss'] = loss_supervised
        if self.do_thresh_penalization and batch_idx >= self.first_batch_pen:
            outputs['pen_loss'] = self.pen_fn()
            outputs['loss'] = loss_supervised + outputs['pen_loss']

        return outputs, predictions
