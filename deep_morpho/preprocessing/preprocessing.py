from abc import ABC

from general.nn.experiments.experiment_methods import ExperimentMethods


class Preprocessor(ExperimentMethods, ABC):
    def __call__(self, *args, **kwargs):
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"