import torch
import matplotlib.pyplot as plt

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
            # img, target = batch[0][0], batch[1][0].unsqueeze(0)
            # # print(preds)
            # pred = preds[0].unsqueeze(0)
            # input_pred_target = torch.cat([img, pred, target], dim=2)
            # trainer.logger.experiment.add_image("preds/input_pred_target", input_pred_target, trainer.global_step)
            img, target = batch[0][0].squeeze(), batch[1][0].squeeze()
            pred = preds[0].squeeze()
            fig = self.plot_three(*[k.cpu().detach() for k in [img, pred, target]])
            trainer.logger.experiment.add_figure("preds/input_pred_target", fig, trainer.global_step)

        self.idx += 1

    def plot_three(self, img, pred, target):
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(img, cmap='gray')
        axs[0].set_title('input')

        axs[1].imshow(pred, cmap='gray')
        axs[1].set_title('pred')

        axs[2].imshow(target, cmap='gray')
        axs[2].set_title('target')

        return fig