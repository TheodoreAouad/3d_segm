import inspect
from typing import Any, Dict, List, Optional

from abc import ABC


class DataModule(ABC):
    @classmethod
    def select_(cls, name: str) -> Optional["DataModule"]:
        """
        Recursive class method iterating over all subclasses to return the
        desired model class.
        """
        if cls.__name__.lower() == name:
            return cls

        for subclass in cls.__subclasses__():
            selected = subclass.select_(name)
            if selected is not None:
                return selected

        return None

    @classmethod
    def select(cls, name: str, **kwargs: Any) -> "DataModule":
        """
        Class method iterating over all subclasses to instantiate the desired
        model.
        """

        selected = cls.select_(name)
        if selected is None:
            raise ValueError("The selected model was not found.")

        return selected

    @classmethod
    def listing(cls) -> List[str]:
        """List all the available models."""
        subclasses = set()
        if not inspect.isabstract(cls):
            subclasses = {cls.__name__.lower()}

        for subclass in cls.__subclasses__():
            subclasses = subclasses.union(subclass.listing())

        return list(subclasses)

    @classmethod
    def default_args(cls) -> Dict[str, dict]:
        """Return the default arguments of the model, in the format of argparse.ArgumentParser
        Ex:
        >>> args = {}"""
        return {
            name: {"default": None}
            for name in inspect.signature(cls.__init__).parameters if name != "self"
        }

    @classmethod
    def listing_subclasses(cls) -> List[str]:
        """List all the available models."""
        subclasses = set()
        if not inspect.isabstract(cls):
            subclasses = {cls}

        for subclass in cls.__subclasses__():
            subclasses = subclasses.union(subclass.listing_subclasses())

        return list(subclasses)
