from .observable_layers import ObservableLayers


class ShowSelemBinary(ObservableLayers):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_selem_and_op = {}

    def on_train_batch_end_with_preds_layers_chans(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        preds,
        layer: "nn.Module",
        layer_idx: int,
        chan_input: int,
        chan_output: int,
    ):
        bise_layer = layer.bises[chan_input]
        with torch.no_grad():
            bise_layer.find_selem_and_operation_chan(chan_output, v1=0, v2=1)
        if not bise_layer._is_activated[chan_output]:
            return

        selem, operation = bise_layer.learned_selem[chan_output], bise_layer.learned_operation[chan_output]
        operation = operation_code_inverse[operation]

        fig = self.selem_fig(selem, operation)
        trainer.logger.experiment.add_figure(f"learned_selem/binary/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}", fig, trainer.global_step)
        self.last_selem_and_op[(layer_idx, chan_input, chan_output)] = (selem, operation)


    @staticmethod
    def selem_fig(selem, operation):
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(selem, interpolation="nearest", vmin=0, vmax=1, cmap="gray")
        plt.title(operation)
        return fig


    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        saved = []

        for (layer_idx, chan_input, chan_output), (selem, operation) in self.last_selem_and_op.items():
            fig = self.selem_fig(selem, operation)
            fig.savefig(join(final_dir, f"layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}.png"))
            plt.close(fig)
            saved.append(fig)

        return saved