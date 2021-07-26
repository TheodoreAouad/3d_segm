import torch
from general.nn.observables import Observable
from general.utils import max_min_norm


class PlotPreds(Observable):

    def __init__(self, freq: int = 100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.idx = 0


    def on_train_batch_end_with_preds(
            self,
            trainer: 'pl.Trainer',
            pl_module: 'pl.LightningModule',
            outputs: 'STEP_OUTPUT',
            batch: 'Any',
            batch_idx: int,
            preds: 'Any',
    ) -> None:
        if self.idx % self.freq == 0:
            img, target = batch[0][0], batch[1][0].unsqueeze(0)
            # print(preds)
            pred = preds[0].unsqueeze(0)
            input_pred_target = torch.cat([img, pred, target], dim=2)
            trainer.logger.experiment.add_image("preds/input_pred_target", input_pred_target, trainer.global_step)
        self.idx += 1
