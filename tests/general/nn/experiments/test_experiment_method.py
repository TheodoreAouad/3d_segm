from general.nn.experiments.experiment_methods import ExperimentMethods


class TestExperimentMethods:
    @staticmethod
    def test_simple_class():
        class A(ExperimentMethods):
            def __init__(self, a=1, b=2):
                super().__init__()

        assert A.default_args() == {"a": {"default": 1}, "b": {"default": 2}}

    @staticmethod
    def test_inheritence1():
        class A(ExperimentMethods):
            def __init__(self, a=1, b=2):
                super().__init__()

        class B(A):
            def __init__(self, c=3, d=4, **kwargs):
                super().__init__()

        assert B.default_args() == {"a": {"default": 1}, "b": {"default": 2}, "c": {"default": 3}, "d": {"default": 4}}

    @staticmethod
    def test_inheritence2():
        class A(ExperimentMethods):
            def __init__(self, a=1, b=2):
                super().__init__()

        class B(A):
            def __init__(self, c=3, d=4, **mm):
                super().__init__()

        class C(B):
            def __init__(self, e=5, f=6, *aa):
                super().__init__()

        assert C.default_args() == {"a": {"default": 1}, "b": {"default": 2}, "c": {"default": 3}, "d": {"default": 4},
                                    "e": {"default": 5}, "f": {"default": 6}}

    @staticmethod
    def test_inheritence3():
        class A(ExperimentMethods):
            def __init__(self, a=1, b=2):
                super().__init__()

        class B(A):
            def __init__(self, c=3, d=4,):
                super().__init__()

        assert B.default_args() == {"c": {"default": 3}, "d": {"default": 4}}

    @staticmethod
    def test_inheritence4():
        class A(ExperimentMethods):
            def __init__(self, a=1, b=2, *args, **kwargs):
                super().__init__()

        assert A.default_args() == {"a": {"default": 1}, "b": {"default": 2}}


    @staticmethod
    def test_inheritence_multi():
        class A(ExperimentMethods):
            def __init__(self, a=1, b=2):
                super().__init__()

        class B(A):
            def __init__(self, a=3, *args):
                super().__init__()

        assert B.default_args() == {"a": {"default": 3}, "b": {"default": 2}}
