from general.nn.observables import Observable


class PlotLUIParameters(Observable):

    def on_train_batch_end_with_preds(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        preds: "Any",
    ) -> None:
        for output_chan in range(pl_module.model.chan_outputs):
            metrics_chan = {}
            for input_chan in range(pl_module.model.chan_inputs):
                metrics_chan[f"beta_in_{input_chan}"] = pl_module.model.weight[output_chan, input_chan]

            trainer.logger.experiment.add_scalars(f'weights/train/beta_chan_{input_chan}', metrics_chan, trainer.global_step)
            trainer.logger.log_metrics({f'weights/train/bias_chan_{input_chan}': pl_module.model.bias[output_chan]}, trainer.global_step)
