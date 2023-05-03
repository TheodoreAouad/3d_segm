import torch

from deep_morpho.models.ste import (
    _STEClippedIdentity, _STEQuantizedK, _STEBernoulli, _STEXNor, _STEDoReFaBinary,
    STEClippedIdentity, STEQuantizedK, STEBernoulli, STEXNor, STEDoReFaBinary,
)


class TestSTEClippedIdentity:

    @staticmethod
    def test_forward():
        layer = STEClippedIdentity()
        x = torch.randn(10, 10)
        y = layer(x)

        assert ((x > 0)[x > 0].float() == y[x > 0]).all()
        assert ((-((x < 0)[x < 0]).float()) == y[x < 0]).all()

    @staticmethod
    def test_backward():
        layer = STEClippedIdentity()
        x = torch.randn(10, 10, requires_grad=True)
        y = layer(torch.clip(x, min=-1, max=1))
        y.sum().backward()

        grad1 = x.grad.clone()
        x.grad.zero_()

        x.sum().backward()
        grad2 = x.grad.clone()

        assert (grad1[(x >= -1) & (x <= 1)] == grad2[(x >= -1) & (x <= 1)]).all()
        assert grad1[x > 1].abs().sum() == 0
        assert grad1[x < -1].abs().sum() == 0


class TestSTEQuantizedK:

    @staticmethod
    def test_forward():
        layer = STEQuantizedK(k=3)
        x = torch.rand(10, 10)
        y = layer(x)
        assert len(y.unique()) == 2 ** 3
        pass

    @staticmethod
    def test_backward():
        layer = STEQuantizedK(k=3)
        x = torch.rand(10, 10, requires_grad=True)
        y = layer(x)
        y.sum().backward()

        grad1 = x.grad.clone()
        x.grad.zero_()

        x.sum().backward()
        grad2 = x.grad.clone()

        assert (grad1 == grad2).all()


class TestSTEBernoulli:
    @staticmethod
    def test_forward():
        layer = STEBernoulli()
        x = torch.rand(10, 10)
        y = layer(x)
        assert len(y.unique()) == 2

    @staticmethod
    def test_backward():
        layer = STEBernoulli()
        x = torch.rand(10, 10, requires_grad=True)
        y = layer(x)
        y.sum().backward()

        grad1 = x.grad.clone()
        x.grad.zero_()

        x.sum().backward()
        grad2 = x.grad.clone()

        assert (grad1 == grad2).all()


class TestSTEXNor:
    @staticmethod
    def test_forward():
        layer = STEXNor()
        x = torch.rand(10, 3, 10, 10)
        layer(x)


    @staticmethod
    def test_backward():
        layer = STEXNor()
        x = torch.rand(10, 3, 10, 10, requires_grad=True)
        y = layer(x)
        y.sum().backward()

        grad1 = x.grad.clone()
        x.grad.zero_()

        x.sum().backward()
        grad2 = x.grad.clone()

        assert (grad1 == grad2).all()


class TestSTEDoReFaBinary:
    @staticmethod
    def test_forward():
        layer = STEDoReFaBinary()
        x = torch.rand(10, 3, 10, 10)
        layer(x)


    @staticmethod
    def test_backward():
        layer = STEDoReFaBinary()
        x = torch.rand(10, 3, 10, 10, requires_grad=True)
        y = layer(x)
        y.sum().backward()

        grad1 = x.grad.clone()
        x.grad.zero_()

        x.sum().backward()
        grad2 = x.grad.clone()

        assert (grad1 == grad2).all()
