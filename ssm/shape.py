import warnings

import numpy as np

from .utils import get_norm_transform, transform_cloud
from .icp import register_icp, nearest_neighbor
from .sample_mesh import dijkstra_sampling


class Shape:

    def __init__(
        self,
        label: str = None,
        volume: np.ndarray = None,
        vertexes: np.ndarray = None,
        faces: np.ndarray = None,
        normals: np.ndarray = None,
        values: np.ndarray = None,
        reference: "Shape" = None,
    ):
        self.label = label
        self.volume = volume
        self.vertexes = vertexes
        self.faces = faces
        self.normals = normals
        self.values = values
        self.reference = reference

        self.Tref: np.ndarray = None
        self.sample: np.ndarray = None
        self.dist_to_sample: np.ndarray = None

    @property
    def Tnorm(self) -> np.ndarray:
        assert self.vertexes is not None
        return get_norm_transform(self.vertexes_mean, self.vertexes_std)

    @property
    def Tnorm_inv(self) -> np.ndarray:
        assert self.vertexes is not None
        return get_norm_transform(self.vertexes_mean, self.vertexes_std, invert=True)

    # TODO: do not compute multiple times the mean if called multiple times
    @property
    def vertexes_mean(self) -> np.ndarray:
        return self.vertexes.mean(0)

    # TODO: do not compute multiple times the mean if called multiple times
    @property
    def vertexes_std(self) -> np.ndarray:
        return self.vertexes.std(0)

    def register_icp_to_reference(
        self,
        reference: "Shape" = None,
        allow_reflection: bool = False,
        init_pose=None,
        max_iterations: int = 1000,
        tolerance: float = 1e-5,
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Performs ICP registration onto ref_verts (Shape = T @ ref)
        Args:
            reference (Shape): Shape to register to.
            allow_reflection (bool): if False, will remove reflections
            max_iterations (int): max number of iterations for ICP algorithm
            init_pose (np.ndarray): initial transform
            tolerance (float): min accepted difference of tansform between two iterations of ICP to converge

        Returns:
            np.ndarray, np.ndarray, int: Transform of size (d+1) x (d+1).
                                        errors of size (N). number of iterations to converge.
        """
        if reference is None:
            reference = self.reference

        T, errs, n_iters = register_icp(
            reference.vertexes, self.vertexes,
            allow_reflection=allow_reflection, init_pose=init_pose, max_iterations=max_iterations, tolerance=tolerance,)
        self.Tref = T

        return self.Tref, errs, n_iters

    def align_samples(self, reference: "Shape" = None) -> (np.ndarray, np.ndarray):
        """ Creates samples alignes with reference's samples.
        """
        if reference is None:
            reference = self.reference

        if self.Tref is None:
            warnings.warn("No registration already done. Performing ICP registration with default parameters...")
            self.register_icp_to_reference(reference)

        _, neigh_sampling = nearest_neighbor(
            transform_cloud(self.Tref, reference.sample),
            self.vertexes
        )

        self.sample = self.vertexes[neigh_sampling]
        return self.sample

    def perform_sampling(self, n_points: int, sampling_fn: str = "dijkstra", verbose: bool = False) -> np.ndarray:
        """ Performs uniform sampling.

        Args:
            n_points (int): number of points in the resulting sample
            sampling_fn (str): algorithm to use to sample
            verbose (bool): shows progress bar

        Returns:
            np.ndarray: (n_points x d) matrix.
        """
        if sampling_fn == "dijkstra":
            self.sample, self.dist_to_sample = dijkstra_sampling(self.vertexes, self.faces, n_points, verbose=verbose)
        else:
            raise NotImplementedError

        return self.sample, self.dist_to_sample

    def set_reference(self, reference: "Shape"):
        self.reference = reference
        return self

    def scatter_cloud(self, ax, **kwargs):
        ax.scatter(*self.vertexes.T, **kwargs)
        return ax

    def plot_trimap(self, ax, **kwargs):
        ax.plot_trimap(*self.vertexes.T, triangles=self.faces, **kwargs)
        return ax

    def plot_sample_distance(self, ax, **kwargs):
        ax.scatter(*self.sample.T, c='r', s=200, **kwargs)
        ax.scatter(*self.vertexes.T, c=self.dist_to_sample, **kwargs)
        return ax

    def plot_compare_meshes(self, ax, **kwargs):
        assert self.faces is not None
        assert self.reference.faces is not None
        Tref_vert = transform_cloud(self.Tref, self.reference.vertexes)
        ax.plot_trimap(*Tref_vert.T, triangles=self.reference.faces, label='ref', **kwargs)
        ax.plot_trimap(*self.vertexes.T, triangles=self.faces, label='shape', **kwargs)
        ax.legend()
        return ax

    def plot_compare_point_cloud(self, ax, **kwargs):
        Tref_vert = transform_cloud(self.Tref, self.reference.vertexes)
        ax.scatter(*Tref_vert.T, label='ref', **kwargs)
        ax.scatter(*self.vertexes.T, label='shape', **kwargs)
        ax.legend()
        return ax

    def plot_compare_samples(self, ax, **kwargs):
        Tsampling = transform_cloud(self.Tref, self.reference.sample)
        ax.scatter(*Tsampling.T, label='ref', **kwargs)
        ax.scatter(*self.sample.T, label='shape', **kwargs)

        for s1, s2 in zip(Tsampling, self.sample):
            toplot = [[s1[i], s2[i]] for i in range(3)]
            ax.plot(*toplot)

        ax.legend()
        return ax
