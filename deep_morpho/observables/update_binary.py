from general.nn.observables import Observable


class UpdateBinary(Observable):
    """Update binary parameters at freq for training batch, and at the end of every training epoch."""

    def __init__(self, freq,):
        super().__init__()
        self.freq = freq

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        if self.freq_idx % self.freq == 0:
            self.apply_update(pl_module)
        self.freq_idx += 1

    def apply_update(self, pl_module):
        init_mode = pl_module.model.binary_mode
        pl_module.model.binary(mode=True, update_binaries=True)  # Updating all binaries
        pl_module.model.binary(mode=init_mode, update_binaries=False)  # Resetting original binary mode

    def on_train_epoch_end(self, trainer, pl_module):
        self.apply_update(pl_module)
