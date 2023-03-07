from typing import Dict
from os.path import join
import pathlib

import numpy as np

from general.nn.observables import Observable
from deep_morpho.models import BiSEBase
from general.utils import save_json
from deep_morpho.models import NotBinaryNN


class ActivatednessObservable(Observable):

    def __init__(self, freq=1, layers=None, layer_name="layers", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last = {}
        self.freq = freq
        self.freq_idx = 0
        self.layers = layers
        self.layer_name = layer_name

    def on_train_epoch_end(self, trainer, pl_module):
        if self.freq is None or self.freq_idx % self.freq != 0:
            self.freq_idx += 1
            return
        self.freq_idx += 1

        layers = self._get_layers(pl_module)


        self.last["all"] = {"n_dilation": 0, "n_erosion": 0, "n_activated": 0, "n_total": 0, }
        for layer_idx, layer in enumerate(layers):
            self.last[layer_idx] = {"n_dilation": 0, "n_erosion": 0, "n_activated": 0, "n_total": 0, }
            for module_ in layer.modules():
                if isinstance(module_, BiSEBase) and not isinstance(module_, NotBinaryNN):
                    is_activated = module_.is_activated
                    self.last[layer_idx]["n_dilation"] += (module_.learned_operation[is_activated] == module_.operation_code["dilation"]).sum()
                    self.last[layer_idx]["n_erosion"] += (module_.learned_operation[is_activated] == module_.operation_code["erosion"]).sum()

                    self.last[layer_idx]["n_activated"] += self.last[layer_idx]["n_dilation"] + self.last[layer_idx]["n_erosion"]
                    self.last[layer_idx]["n_total"] += len(is_activated)

            for key, value in self.last[layer_idx].items():
                self.last["all"][key] += value

            self.last[layer_idx].update({
                "ratio_dilation": self.last[layer_idx]["n_dilation"] / (self.last[layer_idx]["n_total"] + 1e-5),
                "ratio_erosion": self.last[layer_idx]["n_erosion"] / (self.last[layer_idx]["n_total"] + 1e-5),
                "ratio_activated": self.last[layer_idx]["n_activated"] / (self.last[layer_idx]["n_total"] + 1e-5),
            })

            for key, value in self.last[layer_idx].items():
                trainer.logger.experiment.add_scalar(f"activatedness_details/{key}/layer_{layer_idx}", value, trainer.current_epoch)
                # pl_module.log(f"activatedness/layer_{layer_idx}/{key}", value,)


        self.last["all"].update({
            "ratio_dilation": self.last["all"]["n_dilation"] / (self.last["all"]["n_total"] + 1e-5),
            "ratio_erosion": self.last["all"]["n_erosion"] / (self.last["all"]["n_total"] + 1e-5),
            "ratio_activated": self.last["all"]["n_activated"] / (self.last["all"]["n_total"] + 1e-5),
        })

        for key, value in self.last["all"].items():
            trainer.logger.experiment.add_scalar(f"activatedness/all/{key}", value, trainer.current_epoch)
            # pl_module.log(f"activatedness/all/{key}", value,)


    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        dict_str = {}
        for k1, v1 in self.last.items():
            dict_str[f"{k1}"] = {}
            for k2, v2 in v1.items():
                dict_str[f"{k1}"][f"{k2}"] = str(v2)
        save_json(dict_str, join(final_dir, "activatedness.json"))
        return self.last


    def save_hparams(self) -> Dict:
        res = {}
        for layer, dict_key in self.last.items():
            for key, value in dict_key.items():
                res[f"{key}_{layer}"] = value
        return res


    def _get_layers(self, pl_module):

        if self.layers is not None:
            return self.layers

        if hasattr(pl_module.model, self.layer_name):
            return getattr(pl_module.model, self.layer_name)


        raise NotImplementedError('Cannot automatically select layers for model. Give them manually.')


class ClosestDistObservable(Observable):

    def __init__(self, freq=1, layers=None, layer_name="layers", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last = {}
        self.freq = freq
        self.freq_idx = 0
        self.layers = layers
        self.layer_name = layer_name

    def on_train_epoch_end(self, trainer, pl_module):
        if self.freq is None or self.freq_idx % self.freq != 0:
            self.freq_idx += 1
            return
        self.freq_idx += 1

        layers = self._get_layers(pl_module)


        self.last["all"] = {"closest_dist": np.array([]), "n_dilation": 0, "n_erosion": 0, "n_total": 0, "n_closest": 0}
        for layer_idx, layer in enumerate(layers):
            self.last[layer_idx] = {"closest_dist": np.array([]), "n_dilation": 0, "n_erosion": 0, "n_total": 0, "n_closest": 0}
            for module_ in layer.modules():
                if isinstance(module_, BiSEBase) and not isinstance(module_, NotBinaryNN):
                    self.last[layer_idx]["closest_dist"] = np.concatenate([self.last[layer_idx]["closest_dist"], module_.closest_selem_dist])

                    is_activated = module_.is_activated
                    self.last[layer_idx]["n_closest"] += len(is_activated) - is_activated.sum()

                    self.last[layer_idx]["n_dilation"] += (module_.closest_operation[~is_activated] == module_.operation_code["dilation"]).sum()
                    self.last[layer_idx]["n_erosion"] += (module_.closest_operation[~is_activated] == module_.operation_code["erosion"]).sum()

                    self.last[layer_idx]["n_total"] += len(is_activated)

            self.last["all"]["closest_dist"] = np.concatenate([self.last["all"]["closest_dist"], self.last[layer_idx]["closest_dist"]])

            for key, value in self.last[layer_idx].items():
                if key == "closest_dist":
                    continue
                self.last["all"][key] += value

            self.last[layer_idx].update({
                "ratio_dilation": self.last[layer_idx]["n_dilation"] / (self.last[layer_idx]["n_total"] + 1e-5),
                "ratio_erosion": self.last[layer_idx]["n_erosion"] / (self.last[layer_idx]["n_total"] + 1e-5),
                "ratio_closest": self.last[layer_idx]["n_closest"] / (self.last[layer_idx]["n_total"] + 1e-5),
            })

            for key, value in self.last[layer_idx].items():
                if key == "closest_dist":
                    continue
                trainer.logger.experiment.add_scalar(f"closest_details/{key}/layer_{layer_idx}", value, trainer.current_epoch)
                # pl_module.log(f"closest/layer_{layer_idx}/{key}", value,)

            if len(self.last[layer_idx]["closest_dist"]) > 0:
                trainer.logger.experiment.add_histogram(f"closest_details/closest_dist/layer_{layer_idx}", self.last[layer_idx]["closest_dist"], trainer.current_epoch)

            for key, value in self.last["all"].items():
                self.last["all"][key] += value

        self.last["all"].update({
            "ratio_dilation": self.last["all"]["n_dilation"] / (self.last["all"]["n_total"] + 1e-5),
            "ratio_erosion": self.last["all"]["n_erosion"] / (self.last["all"]["n_total"] + 1e-5),
            "ratio_closest": self.last["all"]["n_closest"] / (self.last["all"]["n_total"] + 1e-5),
        })

        for key, value in self.last["all"].items():
            if key == "closest_dist":
                continue
            trainer.logger.experiment.add_scalar(f"closest/all/{key}", value, trainer.current_epoch)
            # pl_module.log(f"activatedness/all/{key}", value,)

        if len(self.last["all"]["closest_dist"]) > 0:
            trainer.logger.experiment.add_histogram(f"closest/all/closest_dist", self.last["all"]["closest_dist"], trainer.current_epoch)

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        dict_str = {}
        for k1, v1 in self.last.items():
            dict_str[f"{k1}"] = {}
            for k2, v2 in v1.items():
                if k2 == "closest_dist":
                    continue
                dict_str[f"{k1}"][f"{k2}"] = str(v2)
        save_json(dict_str, join(final_dir, "closest_dist.json"))
        return self.last


    def save_hparams(self) -> Dict:
        res = {}
        for layer, dict_key in self.last.items():
            for key, value in dict_key.items():
                if key == "closest_dist":
                    continue
                res[f"{key}_{layer}"] = value
        return res


    def _get_layers(self, pl_module):

        if self.layers is not None:
            return self.layers

        if hasattr(pl_module.model, self.layer_name):
            return getattr(pl_module.model, self.layer_name)


        raise NotImplementedError('Cannot automatically select layers for model. Give them manually.')
