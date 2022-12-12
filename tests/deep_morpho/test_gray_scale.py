import torch
import numpy as np
from skimage import morphology as morp
import pytest

from deep_morpho.gray_scale import level_sets_from_gray, gray_from_level_sets


class TestLevelSetsFromGray():

    @staticmethod
    def test_inverse():
        img = np.random.randint(0, 10, (20, 20))
        img2 = gray_from_level_sets(*level_sets_from_gray(img))
        assert np.sum(np.abs(img - img2)) == 0

    @staticmethod
    def test_inverse_values():
        img = np.random.randint(0, 10, (20, 20))
        values = range(0, 20)
        img2 = gray_from_level_sets(*level_sets_from_gray(img, values))
        assert np.sum(np.abs(img - img2)) == 0

    @staticmethod
    def test_inverse_torch_cpu():
        img = torch.randint(0, 10, (20, 20))
        img2 = gray_from_level_sets(*level_sets_from_gray(img))
        assert torch.sum(torch.abs(img - img2)) == 0

    @staticmethod
    def test_inverse_values_torch_cpu_run():
        img = torch.randint(0, 10, (20, 20))
        values = range(0, 20)
        img2 = gray_from_level_sets(*level_sets_from_gray(img, values))
        assert torch.sum(torch.abs(img - img2)) == 0

    @staticmethod
    def test_inverse_torch_gpu():
        img = torch.randint(0, 10, (20, 20)).to("cuda")
        img2 = gray_from_level_sets(*level_sets_from_gray(img))
        assert torch.sum(torch.abs(img - img2)) == 0

    @staticmethod
    def test_inverse_torch_gpu_values_run():
        img = torch.randint(0, 10, (20, 20)).to("cuda")
        values = range(0, 20)
        img2 = gray_from_level_sets(*level_sets_from_gray(img, values))
        assert torch.sum(torch.abs(img - img2)) == 0


    @staticmethod
    @pytest.mark.parametrize("operation", [morp.erosion, morp.dilation, morp.opening, morp.closing])
    def test_morpop_gray_from_binary(operation):
        img = np.random.randint(0, 10, (20, 20))
        selem = morp.disk(1)
        ls, values = level_sets_from_gray(img)

        for idx in range(ls.shape[0]):
            ls[idx] = operation(ls[idx], selem)

        dil1 = gray_from_level_sets(ls, values)
        dil2 = operation(img, selem)

        assert np.sum(np.abs(dil1 - dil2)) == 0

    @staticmethod
    def test_undersampling():
        img = np.random.randint(0, 10, (20, 20))
        ls, values = level_sets_from_gray(img, n_values=5)
        assert len(values) == 5
        assert ls.shape[0] == 5