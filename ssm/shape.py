import warnings
from typing import List, Tuple

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

from .utils import get_norm_transform, transform_cloud
from .icp import register_icp, nearest_neighbor
from .sample_mesh import dijkstra_sampling, dijkstra_mesh


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
        """
        Initialize the Sample object.

        Args:
            self: write your description
            label: write your description
            volume: write your description
            np: write your description
            ndarray: write your description
            vertexes: write your description
            np: write your description
            ndarray: write your description
            faces: write your description
            np: write your description
            ndarray: write your description
            normals: write your description
            np: write your description
            ndarray: write your description
            values: write your description
            np: write your description
            ndarray: write your description
            reference: write your description
        """
        self.label = label
        self.volume = volume
        self.vertexes = vertexes
        self.faces = faces
        self.normals = normals
        self.values = values
        self.reference = reference

        self.sample_idx = None
        self.Tref: np.ndarray = None
        self.dist_to_sample: np.ndarray = None
        self.Tprocrustes: np.ndarray = None

    @property
    def Tnorm(self) -> np.ndarray:
        """
        The T - norm of the current mesh.

        Args:
            self: write your description
        """
        assert self.vertexes is not None
        return get_norm_transform(self.vertexes_mean, self.centered_norm)

    @property
    def Tnorm_inv(self) -> np.ndarray:
        """
        Inverted inverse of Tnorm.

        Args:
            self: write your description
        """
        assert self.vertexes is not None
        return get_norm_transform(self.vertexes_mean, self.centered_norm, invert=True)

    # TODO: do not compute multiple times the mean if called multiple times
    @property
    def vertexes_mean(self) -> np.ndarray:
        """
        The mean of the vertexes in the graph.

        Args:
            self: write your description
        """
        return self.vertexes.mean(0)

    # TODO: do not compute multiple times the mean if called multiple times
    @property
    def vertexes_std(self) -> np.ndarray:
        """
        Return the standard deviation of all vertexes.

        Args:
            self: write your description
        """
        return self.vertexes.std(0)

    @property
    def centered_norm(self) -> float:
        """
        Return the norm of the center of the mesh.

        Args:
            self: write your description
        """
        return np.linalg.norm(self.vertexes - self.vertexes_mean)

    @property
    def sample(self) -> np.ndarray:
        """
        Sample of the current mesh.

        Args:
            self: write your description
        """
        return self.vertexes[self.sample_idx]

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

    def match_samples(self, reference: "Shape" = None) -> (np.ndarray, np.ndarray):
        """ Creates samples matches with reference's samples.
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

        # self.sample = self.vertexes[neigh_sampling]
        self.sample_idx = neigh_sampling
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
            self.sample_idx, self.dist_to_sample = dijkstra_sampling(self.vertexes, self.faces, n_points, verbose=verbose)
        else:
            raise NotImplementedError

        return self.sample, self.dist_to_sample

    def set_reference(self, reference: "Shape"):
        """
        Set the reference for this entity.

        Args:
            self: write your description
            reference: write your description
        """
        self.reference = reference
        return self

    def scatter_cloud(self, ax, **kwargs):
        """
        Scatter plot of the cloud.

        Args:
            self: write your description
            ax: write your description
        """
        ax.scatter(*self.vertexes.T, **kwargs)
        return ax

    def plot_trimap(self, ax, **kwargs):
        """
        Plot the trimap of the mesh.

        Args:
            self: write your description
            ax: write your description
        """
        ax.plot_trimap(*self.vertexes.T, triangles=self.faces, **kwargs)
        return ax

    def plot_sample_distance(self, ax, **kwargs):
        """
        Plot the sample distances.

        Args:
            self: write your description
            ax: write your description
        """
        ax.scatter(*self.sample.T, c='r', s=200, **kwargs)
        ax.scatter(*self.vertexes.T, c=self.dist_to_sample, **kwargs)
        return ax

    def plot_compare_meshes(self, ax, **kwargs):
        """
        Plot comparison of meshes.

        Args:
            self: write your description
            ax: write your description
        """
        assert self.faces is not None
        assert self.reference.faces is not None
        Tref_vert = transform_cloud(self.Tref, self.reference.vertexes)
        ax.plot_trimap(*Tref_vert.T, triangles=self.reference.faces, label='ref', **kwargs)
        ax.plot_trimap(*self.vertexes.T, triangles=self.faces, label='shape', **kwargs)
        ax.legend()
        return ax

    def plot_compare_point_cloud(self, ax, show_sampling=False, **kwargs):
        """
        Plot the cloud of the comparison point cloud.

        Args:
            self: write your description
            ax: write your description
            show_sampling: write your description
        """
        Tref_vert = transform_cloud(self.Tref, self.reference.vertexes)
        ax.scatter(*Tref_vert.T, label='ref', **kwargs)
        ax.scatter(*self.vertexes.T, label='shape', **kwargs)

        if show_sampling:
            Tsampling = transform_cloud(self.Tref, self.reference.sample)
            self.plot_link_samples(ax, Tsampling, self.sample)

        ax.legend()
        return ax

    def plot_compare_samples(self, ax, **kwargs):
        """
        Plot samples that are different from the reference sample

        Args:
            self: write your description
            ax: write your description
        """
        Tsampling = transform_cloud(self.Tref, self.reference.sample)
        ax.scatter(*Tsampling.T, label='ref', c='g', **kwargs)
        ax.scatter(*self.sample.T, label='shape', c='r', **kwargs)

        # for s1, s2 in zip(Tsampling, self.sample):
        #     toplot = [[s1[i], s2[i]] for i in range(3)]
        #     ax.plot(*toplot)
        self.plot_link_samples(ax, Tsampling, self.sample)

        ax.legend()
        return ax

    @staticmethod
    def plot_link_samples(ax, samples1, samples2, **kwargs):
        """
        Plot the link of samples.

        Args:
            ax: write your description
            samples1: write your description
            samples2: write your description
        """
        for s1, s2 in zip(samples1, samples2):
            toplot = [[s1[i], s2[i]] for i in range(3)]
            ax.plot(*toplot, **kwargs)
        return ax

    def apply_procrustes(self, reference):
        """
        Applies the procrustes to the reference.

        Args:
            self: write your description
            reference: write your description
        """
        pass

    def align_samples(self, reference: "Shape" = None) -> np.ndarray:
        """ Given a reference sample, align the shape sample using procrustes anaysis.
        """
        if reference is None:
            reference = self.reference
        nsample = transform_cloud(self.Tnorm, self.sample)
        ref_nsample = transform_cloud(reference.Tnorm, reference.sample)
        R, scale = orthogonal_procrustes(nsample, ref_nsample)

        T = np.zeros((4, 4))
        T[:3, :3] = R * scale
        T[-1, -1] = 1
        T_to_ref = reference.Tnorm_inv @ T

        sample_on_ref = transform_cloud(T_to_ref, nsample)

        return sample_on_ref

    def dijkstra_to_sample(self):
        """
        The distance from the sampled vertices to the nearest vertex.

        Args:
            self: write your description
        """
        self.dist_to_sample = dijkstra_mesh(self.vertexes, self.faces, self.sample_idx)
        return self.dist_to_sample


class SSM:

    def __init__(self, shapes: List[Shape] = None, reference: Shape = None):
        """
        Initializes the PCA object.

        Args:
            self: write your description
            shapes: write your description
            reference: write your description
        """
        self.shapes = shapes
        self.reference = reference

        self.pca = None
        self.pca_basis: List[Shape] = None

    def compute_pca(self):
        """
        Compute the PCA.

        Args:
            self: write your description
        """
        self.pca = PCA(n_components=len(self.shapes))
        self.pca.fit(self.all_samples.reshape(len(self), -1))

    @property
    def all_samples(self) -> np.ndarray:
        """
        Returns the samples in all shapes as a 1D numpy array.

        Args:
            self: write your description
        """
        return np.stack([shape.sample for shape in self.shapes], axis=0)

    def __len__(self):
        """
        Returns the number of shapes in the mesh.

        Args:
            self: write your description
        """
        return len(self.shapes)

    def get_component(self, idx: int) -> np.ndarray:
        """
        Return the component at the given index

        Args:
            self: write your description
            idx: write your description
        """
        return self.pca.components_[idx].reshape(*self.size)

    def in_pca_basis(self, coords: np.ndarray) -> np.ndarray:
        """
        Return the coordinates of the atoms in the basis.

        Args:
            self: write your description
            coords: write your description
            np: write your description
            ndarray: write your description
        """
        return (coords @ self.pca.components_[:len(coords)]).reshape(*self.size)

    @property
    def size(self) -> Tuple:
        """
        Return the sample size.

        Args:
            self: write your description
        """
        return self.shapes[0].sample.shape
