from typing import Dict, Callable, List

from .dilation_layer import DilationLayer
from general.nn.pytorch_lightning_module.obs_lightning_module import NetLightning
from ..loss import ThresholdPenalization


class LightningDilationLayer(NetLightning):

    def __init__(
        self,
        model_args: Dict,
        learning_rate: float,
        optimizer: Callable,
        output_dir: str,
        loss: Callable,
        do_thresh_penalization: bool = True,
        args_thresh_penalization: Dict = {
            'coef': .5,
            'degree': 2,
            'detach_weights': True,
        },
        first_batch_pen: int = 100,
        optimizer_args: Dict = {},
        observables: [List["Observable"]] = [],
    ):
        super().__init__(
            model=DilationLayer(**model_args),
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
                dilation_layers=[self.model], **self.args_thresh_penalization
            )
        self.save_hyperparameters()

    def obs_training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x).squeeze()

        outputs = {}

        loss_supervised = self.loss(predictions, y)
        outputs['loss_supervised'] = loss_supervised
        outputs['loss'] = loss_supervised
        if self.do_thresh_penalization and batch_idx >= self.first_batch_pen:
            outputs['pen_loss'] = self.pen_fn()
            outputs['loss'] = loss_supervised + outputs['pen_loss']
            self.log('loss/train_pen', outputs['pen_loss'])

        self.log('loss/train_loss', outputs['loss'])
        self.log('loss/train_supervised', outputs['loss_supervised'])

        return outputs, predictions