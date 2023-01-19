from functools import reduce

from pytorch_lightning import LightningModule
from typing import Any, List, Optional, Callable, Dict, Union, Tuple
from ..observables.observable import Observable
from pytorch_lightning.utilities.types import EPOCH_OUTPUT



class ObsLightningModule(LightningModule):

    def __init__(self, observables=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observables: Optional[List[Observable]] = observables

    def training_step(self, batch: Any, batch_idx: int):
        outputs, preds = self.obs_training_step(batch, batch_idx)

        for obs in self.observables:
            if isinstance(obs, Observable):
                obs.on_train_batch_end_with_preds(
                    self.trainer,
                    self.trainer.lightning_module,
                    outputs,
                    batch,
                    batch_idx,
                    preds
                )
        return outputs

    def validation_step(self, batch: Any, batch_idx: int):
        outputs, preds = self.obs_validation_step(batch, batch_idx)

        for obs in self.observables:
            if isinstance(obs, Observable):
                obs.on_validation_batch_end_with_preds(
                    self.trainer,
                    self.trainer.lightning_module,
                    outputs,
                    batch,
                    batch_idx,
                    preds
                )
        return outputs

    def test_step(self, batch: Any, batch_idx: int):
        outputs, preds = self.obs_test_step(batch, batch_idx)

        for obs in self.observables:
            if isinstance(obs, Observable):
                obs.on_test_batch_end_with_preds(
                    self.trainer,
                    self.trainer.lightning_module,
                    outputs,
                    batch,
                    batch_idx,
                    preds
                )

        return outputs

    def test_epoch_end(self, outputs: EPOCH_OUTPUT):
        self.obs_test_epoch_end(outputs)
        for obs in self.observables:
            obs.on_test_epoch_end(self.trainer, self.trainer.lightning_module)

    def obs_training_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def obs_validation_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def obs_test_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def obs_test_epoch_end(self, outputs: EPOCH_OUTPUT):
        pass


class NetLightning(ObsLightningModule):
    def __init__(
            self,
            model: "nn.Module",
            learning_rate: float,
            loss: Union[Callable, Tuple[Callable, Dict], Dict[str, Union[Callable, Tuple[Callable, Dict]]]],
            optimizer: Callable,
            optimizer_args: Dict = {},
            observables: Optional[List[Observable]] = [],
            reduce_loss_fn: Callable = lambda x: reduce(lambda a, b: a + b, x),
            **kwargs
    ):
        """ Basic class for a neural network framework using pytorch lightning, with the predictions available
        for the callbacs.

        Args:
            model (nn.Module): the module we want to train
            learning_rate (float): learning rate of the optimizer
            loss : if a callable is given, loss will be this callable. We can also give a constructor and the arguments.
                If the constructor is given, then the model will also be given as argument. This can be used for losses
                using arguments of the model (e.g. regularization loss on weights). Finally, we can also give a
                dictionary if multiple losses are needed.
                >>> loss = nn.MSELoss()
                >>> loss = (nn.MSELoss, {size_average=True})
                >>> loss = {"mse": (nn.MSELoss, {size_average=True}), "regu": (MyReguLoss, {"lambda_": lambda_})}
            optimizer (Callable): the constructor of the optimizer used to upgrade the parameters using their gradient
            optimizer_args (dict): the init arguments of the constructor of the optimizer, minus the learning rate
            observables (list): list of observables that we want to follow
            reduce_loss_fn (Callable): if we have multiple loss, tells how we want to aggregate all losses
        """

        super().__init__(observables, **kwargs)
        self.model = model
        self.learning_rate = learning_rate
        self.loss = self.configure_loss(loss)
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.reduce_loss_fn = reduce_loss_fn
        # self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_loss(self, loss):
        if isinstance(loss, dict):
            return {k: self.configure_loss(v) for (k, v) in loss.items()}

        if isinstance(loss, tuple):
            return loss[0](model=self.model, **loss[1])

        return loss

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.learning_rate, **self.optimizer_args)

    def obs_training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)

        outputs = self.compute_loss(state="training", ypred=predictions, ytrue=y)

        # return {'loss': loss}, predictions
        return outputs, predictions

    def obs_validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)

        outputs = self.compute_loss(state="validation", ypred=predictions, ytrue=y)

        # return {'val_loss': loss}, predictions
        return outputs, predictions

    def obs_test_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)

        outputs = self.compute_loss(state="test", ypred=predictions, ytrue=y)

        # return {'test_loss': loss}, predictions
        return outputs, predictions

    def compute_loss(self, state, ypred, ytrue):
        values = {}
        assert not ypred.isnan().any(), "NaN in prediction"
        if isinstance(self.loss, dict):
            for key, loss_fn in self.loss.items():
                values[key] = loss_fn(ypred, ytrue)

            if "loss" in self.loss.keys():
                i = 0
                while f"loss_{i}" in self.loss.keys():
                    i += 1
                values[f"loss_{i}"] = values["loss"]

            values["loss"] = self.reduce_loss_fn(values.values())

        else:
            values["loss"] = self.loss(ypred, ytrue)

        grad_values = {}
        for key, value in values.items():
            self.log(f"loss/{state}/{key}", value.item())  # put .item() to avoid memory leak
            if key == "loss":
                continue

            if value.requires_grad:
                value.retain_grad()  # see graph of each loss term
                grad_values[f'{key}_grad'] = value.grad
                values[key] = values[key].detach()

        values['grads'] = grad_values

        return values
