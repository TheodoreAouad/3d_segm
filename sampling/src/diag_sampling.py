import numpy as np

from sampling.src.geometry.entities import StraightCube, Plane


def diag_sampling_annot(true_seg, vecns):
    """
    Samples a segment of the input signal using a diagonal cube.

    Args:
        true_seg: write your description
        vecns: write your description
    """
    markers = np.zeros(true_seg.shape)
    annots = np.zeros_like(true_seg)
    cov = np.zeros_like(true_seg).astype(np.uint8)

    cube = StraightCube(size=true_seg.shape)

    for vecn in vecns.T:
        samples, Xs, Ys = cube.sample_centered_plane(200, (200, 200), Plane(vecn))
        for i in range(3):
            samples[i, samples[i, :] > true_seg.shape[i] - 1] = true_seg.shape[i] - 1
            samples[i, samples[i, :] < 0] = 0
        samples = samples.astype(int)

        cov[samples[0], samples[1], samples[2]] = 1
        annots[samples[0], samples[1], samples[2]] = true_seg[samples[0], samples[1], samples[2]]
        markers[samples[0], samples[1], samples[2]] = np.where(true_seg, 1, -1)[samples[0], samples[1], samples[2]]

    return cov, markers, annots
