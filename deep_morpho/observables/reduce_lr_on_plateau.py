from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import Callback


class BatchReduceLrOnPlateau(Callback):

    def __init__(
        self,
        mode='min',
        factor=0.1,
        patience=10,
        threshold=0.0001,
        threshold_mode='rel',
        cooldown=0,
        min_lr=0,
        eps=1e-08,
        verbose=False,
        on_train=True,
    ):
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.verbose = verbose
        self.on_train = on_train

        self.scheduler = None


    @property
    def kwargs(self):
        return {
            "mode": self.mode,
            "factor": self.factor,
            "patience": self.patience,
            "threshold": self.threshold,
            "threshold_mode": self.threshold_mode,
            "cooldown": self.cooldown,
            "min_lr": self.min_lr,
            "eps": self.eps,
            "verbose": self.verbose,
        }

    def on_train_start(self, trainer, pl_module):
        self.scheduler = ReduceLROnPlateau(optimizer=pl_module.optimizers(), **self.kwargs)

    def on_train_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        if self.on_train:
            self.scheduler.step(outputs['loss'])

    def save(self, save_path: str):
        pass
