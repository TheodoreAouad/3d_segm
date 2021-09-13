from torch.utils.tensorboard.summary import custom_scalars

from .observable_layers import ObservableLayers


class CheckMorpOperation(ObservableLayers):

    def __init__(self, selems, operations, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selems = selems
        self.operations = operations

    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        layers = self._get_layers(pl_module)
        default_layout = {}
        for layer_idx, layer in enumerate(layers):
            tags_dilation = [
                f'comparative/weights/bias_{layer_idx}/bias',
                f'comparative/weights/bias_{layer_idx}/dilation_lb',
                f'comparative/weights/bias_{layer_idx}/dilation_ub'
            ]
            tags_erosion = [
                f'comparative/weights/bias_{layer_idx}/bias',
                f'comparative/weights/bias_{layer_idx}/erosion_lb',
                f'comparative/weights/bias_{layer_idx}/erosion_ub'
            ]
            default_layout.update({
                    f"dilation_{layer_idx}": ['Margin', tags_dilation],
                    f"erosion_{layer_idx}": ['Margin', tags_erosion],
            })

        layout = {"default": default_layout}
        trainer.logger.experiment._get_file_writer().add_summary(custom_scalars(layout))


    def on_train_batch_end_layers(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
        layer: "nn.Module",
        layer_idx: int,
    ):
        erosion_lb, erosion_ub = layer.bias_bounds_erosion(self.selems[layer_idx])
        dilation_lb, dilation_ub = layer.bias_bounds_dilation(self.selems[layer_idx])

        # trainer.logger.experiment.add_scalars(f"comparative/weights/bias_{layer_idx}", {
        #     "bias": layer.bias,
        #     f"erosion_lb": layer.bias - 1/2, #erosion_lb,
        #     f"erosion_ub": layer.bias + 1/2, #erosion_ub,
        #     f"dilation_lb": layer.bias - 1/2, #dilation_lb,
        #     f"dilation_ub": layer.bias + 1/2, #dilation_ub,
        # }, trainer.global_step)
        trainer.logger.experiment.add_scalar(f"comparative/weights/bias_{layer_idx}/bias", -layer.bias, trainer.global_step)
        trainer.logger.experiment.add_scalar(f"comparative/weights/bias_{layer_idx}/erosion_lb", erosion_lb, trainer.global_step)
        trainer.logger.experiment.add_scalar(f"comparative/weights/bias_{layer_idx}/erosion_ub", erosion_ub, trainer.global_step)
        trainer.logger.experiment.add_scalar(f"comparative/weights/bias_{layer_idx}/dilation_lb", dilation_lb, trainer.global_step)
        trainer.logger.experiment.add_scalar(f"comparative/weights/bias_{layer_idx}/dilation_ub", dilation_ub, trainer.global_step)

        if self.operations[layer_idx] == 'dilation':
            metrics = {
                f"metrics/bias - lb(op)_{layer_idx}": -layer.bias - dilation_lb,
                f"metrics/ub(op) - bias_{layer_idx}": dilation_ub - (-layer.bias),
            }
        elif self.operations[layer_idx] == 'erosion':
            metrics = {
                f"metrics/bias - lb(op)_{layer_idx}": -layer.bias - erosion_lb,
                f"metrics/ub(op) - bias_{layer_idx}": erosion_ub - (-layer.bias),
            }
        else:
            raise NotImplementedError('operation must be dilation or erosion.')

        trainer.logger.log_metrics(metrics, trainer.global_step)
