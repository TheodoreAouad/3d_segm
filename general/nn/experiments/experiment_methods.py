import inspect
import warnings
from typing import Any, Dict, List, Optional
from abc import ABC
from enum import EnumMeta


class ExperimentMethods(ABC):
    # Enforce signature of `__init__` method not to have `*args` or `**kwargs` for smooth recursiveness
    # in `default_args` method.
    def __init__(self):
        super().__init__()

    @classmethod
    def select_(cls, name: str) -> Optional["cls"]:
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
    def select(cls, name: str, **kwargs: Any) -> "cls":
        """
        Class method iterating over all subclasses to instantiate the desired
        child.
        """
        name = name.lower()
        selected = cls.select_(name)
        if selected is None:
            err_msg = (""
            f"The selected child {name} was not found for {cls}.\n"
            f"Available children are: {cls.listing()}"
            "")
            raise ValueError(err_msg)

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
    def listing_subclasses(cls) -> List[str]:
        """List all the available models."""
        subclasses = set()
        if not inspect.isabstract(cls):
            subclasses = {cls}

        for subclass in cls.__subclasses__():
            subclasses = subclasses.union(subclass.listing_subclasses())

        return list(subclasses)

    # TODO: if *args or **kwargs present, go see the parent class arguments
    @classmethod
    def default_args(cls) -> Dict[str, dict]:
        """Return the default arguments of the model, in the format of argparse.ArgumentParser"""
        res = {}
        for name, p in inspect.signature(cls.__init__).parameters.items():
            # if name in ["args", "kwargs"]:
            if p.kind in [inspect._ParameterKind.VAR_POSITIONAL, inspect._ParameterKind.VAR_KEYWORD]:
                if len(cls.__bases__) > 1:
                    warnings.warn(""
                        "Multiple inheritance is not supported."
                        "Only the first parent class arguments will be transmitted."
                    "")
                parent_class = cls.__bases__[0]
                res.update({k: v for k, v in parent_class.default_args().items() if k not in res})

            elif name != "self":
                param_dict = {"default": p.default}

                if isinstance(p.default, (bool, int, float, str)):
                    param_dict["type"] = type(p.default)

                if p.annotation is not None:
                    if p.annotation in [bool, int, float, str]:
                        param_dict["type"] = p.annotation

                    elif isinstance(p.annotation, EnumMeta):
                        param_dict["type"] = cls.enum_to_str_fn(p.annotation)

                    # elif p.annotation == List:

                # elif p.annotation != inspect._empty:
                #     param_dict["type"] = p.annotation

                res[name] = param_dict

        return res


    @staticmethod
    def enum_to_str_fn(enum: EnumMeta) -> str:
        """Returns a function that converts the enum to a string."""
        def enum_to_str(string) -> str:
            return enum[string]
        return enum_to_str


    # @staticmethod
    # def str_to_list(string: str) -> List[str]:
    #     """Convert a string to a list of strings."""
    #     return [s.strip() for s in string.split(",") if s.strip() != ""]