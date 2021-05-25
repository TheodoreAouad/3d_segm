import numpy as np
import skimage.morphology as morp

from general.array_morphology import array_dilation, array_erosion


class TestArrayErosion:

    @staticmethod
    def test_array_erosion_2d_rod():
        seg2 = np.zeros((100, 100))
        seg2[47:53, 20:80] = 1

        selem = np.random.randint(0, 2, size=(3, 3))

        c1 = array_erosion(seg2, selem)
        c2 = morp.erosion(seg2, selem)

        assert (c1 != c2).sum() == 0

    @staticmethod
    def test_array_erosion_2d_random():
        seg2 = np.zeros((100, 100))
        seg2[1:-1, 1:-1] = np.random.randint(0, 2, size=(98, 98))

        selem = np.random.randint(0, 2, size=(3, 3))

        c1 = array_erosion(seg2, selem)
        c2 = morp.erosion(seg2, selem)

        assert (c1 != c2).sum() == 0

    @staticmethod
    def test_array_erosion_3d_random():
        seg3 = np.zeros((100, 100, 100))
        seg3[1:-1, 1:-1, 1:-1] = np.random.randint(0, 2, size=(98, 98, 98))

        selem = np.random.randint(0, 2, size=(3, 3, 3))

        c1 = array_erosion(seg3, selem)
        c2 = morp.erosion(seg3, selem)

        assert (c1 != c2).sum() == 0


class TestArrayDilation:

    @staticmethod
    def test_array_dilation_2d_rod():
        seg2 = np.zeros((100, 100))
        seg2[47:53, 20:80] = 1

        selem = np.random.randint(0, 2, size=(3, 3))

        c1 = array_dilation(seg2, selem)
        c2 = morp.dilation(seg2, selem)

        assert (c1 != c2).sum() == 0

    @staticmethod
    def test_array_dilation_2d_random():
        seg2 = np.zeros((100, 100))
        seg2[1:-1, 1:-1] = np.random.randint(0, 2, size=(98, 98))

        selem = np.random.randint(0, 2, size=(3, 3))

        c1 = array_dilation(seg2, selem)
        c2 = morp.dilation(seg2, selem)

        assert (c1 != c2).sum() == 0

    @staticmethod
    def test_array_dilation_3d_random():
        seg3 = np.zeros((100, 100, 100))
        seg3[1:-1, 1:-1, 1:-1] = np.random.randint(0, 2, size=(98, 98, 98))

        selem = np.random.randint(0, 2, size=(3, 3, 3))

        c1 = array_dilation(seg3, selem)
        c2 = morp.dilation(seg3, selem)

        assert (c1 != c2).sum() == 0
