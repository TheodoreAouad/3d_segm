from enum import Enum
from typing import Dict, Any, Tuple
from os.path import join
import pathlib

import torch

from general.utils import save_json
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class ReasonCodeEnum(Enum):
    CHECK_FINITE = 0
    STOPPING_THRESHOLD = 1
    DIVERGENCE_THRESHOLD = 2
    STOP_IMPROVING = 3



class BatchEarlyStopping(EarlyStopping):

    def __init__(self, name: str = '', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.reason_code = None
        self.stopped_batch = None

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        return

    def on_validation_end(self, trainer, pl_module) -> None:
        return

    def on_train_batch_end(self, trainer: 'pl.Trainer', *args, **kwargs) -> None:
        if self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)

    def on_save_checkpoint(self, *args, **kwargs):
        res = super().on_save_checkpoint(None, None, None)  # lightning necessity to avoid *args and kwargs. May change in future versions.
        res['stopped_batch'] = self.stopped_batch
        return res

    def on_load_checkpoint(self, callback_state: Dict[str, Any]) -> None:
        super().on_load_checkpoint(callback_state)
        self.stopped_batch = callback_state['stopped_batch']

    def save(self, save_path: str):
        final_dir = join(save_path, join(self.__class__.__name__, self.name))
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        best_score = self.best_score
        if isinstance(best_score, torch.Tensor):
            best_score = best_score.item()

        to_save = {
            "stopped_batch": self.stopped_batch,
            "wait_count": self.wait_count,
            "best_score": best_score,
            "patience": self.patience,
            "monitor": self.monitor,
            "stopping_reason": str(self.reason_code),
        }

        save_json(to_save, join(final_dir, "results.json"))
        return to_save


    def _run_early_stopping_check(self, trainer) -> None:
        """
        Checks whether the early stopping condition is met
        and if so tells the trainer to stop the training.
        """
        logs = trainer.logged_metrics
        # logs = trainer.callback_metrics

        if (
            trainer.fast_dev_run  # disable early_stopping with fast_dev_run
            or not self._validate_condition_metric(logs)  # short circuit if metric not present
        ):
            return

        current = logs.get(self.monitor)

        # when in dev debugging
        trainer.dev_debugger.track_early_stopping_history(self, current)

        should_stop, reason, self.reason_code = self._evalute_stopping_criteria(current)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.training_type_plugin.reduce_boolean_decision(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
            self.stopped_batch = trainer.global_step
        if reason and self.verbose:
            self._log_info(trainer, reason)

    def _evalute_stopping_criteria(self, current: torch.Tensor) -> Tuple[bool, str]:
        should_stop = False
        reason_str = None
        reason_code = None
        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason_str = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling Trainer to stop."
            )
            reason_code = ReasonCodeEnum.CHECK_FINITE
        elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            should_stop = True
            reason_str = (
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
                " Signaling Trainer to stop."
            )
            reason_code = ReasonCodeEnum.STOPPING_THRESHOLD
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            should_stop = True
            reason_str = (
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to stop."
            )
            reason_code = ReasonCodeEnum.DIVERGENCE_THRESHOLD
        elif self.monitor_op(current - self.min_delta, self.best_score):
            should_stop = False
            reason_str = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason_str = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."
                )
                reason_code = ReasonCodeEnum.STOP_IMPROVING

        return should_stop, reason_str, reason_code
