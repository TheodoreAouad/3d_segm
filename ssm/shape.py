import warnings
from typing import List, Tuple, Dict
from os.path import join, exists
import random

import numpy as np
import networkx as nx
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

from general.utils import uniform_sampling_bound, max_min_norm, colormap_1d
from general.open3d_utils import numpy_to_o3d_mesh, get_o3d_pcd_colored
from .utils import get_norm_transform, transform_cloud, is_outlier_1d, array_wo_idx
from .icp import register_icp, nearest_neighbor
from .hungarian_icp import perfect_matching, register_icp_hungarian
from .sample_mesh import dijkstra_sampling, dijkstra_mesh, create_mesh_graph


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
        origin_path: str = None,
        **kwargs
    ):
        self.label = label
        self.volume = volume
        self.vertexes = vertexes
        self.faces = faces
        self.normals = normals
        self.values = values
        self.reference = reference
        self.origin_path = origin_path

        self.sample_idx = None
        self.Tref: np.ndarray = None
        self.dist_to_sample: np.ndarray = None
        self.closest_sample_point: np.ndarray = None
        self.Tprocrustes: np.ndarray = None
        self.faces_sample: np.ndarray = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def vertices(self) -> np.ndarray:
        return self.vertexes

    @property
    def triangles(self) -> np.ndarray:
        return self.faces

    @staticmethod
    def load_from_path(path: str, label: str = None, ext: str = '.npy', **kwargs) -> "Shape":
        """ Create a Shape from a directory.

        The function will look for 'volume', 'vertexes', 'faces', 'normals', 'values'
        in the directory, with the appropriate extension, and load them.

        Other initialization args can also be given as kwargs, and has priority on the files
        (e.g. if vertexes are given in kwargs, the file vertexes.npy will not be read).
        """
        init_args = dict(
            **{"label": label, 'origin_path': path},
            **kwargs
        )
        for arg in {'volume', 'vertexes', 'faces', 'normals', 'values'}.difference(kwargs.keys()):
            path_arg = join(path, f'{arg}{ext}')
            if exists(path_arg):
                init_args[arg] = np.load(path_arg)
        return Shape(**init_args)

    @property
    def Tnorm(self) -> np.ndarray:
        assert self.vertexes is not None
        return get_norm_transform(self.vertexes_mean, self.centered_norm)

    @property
    def Tnorm_inv(self) -> np.ndarray:
        assert self.vertexes is not None
        return get_norm_transform(self.vertexes_mean, self.centered_norm, invert=True)

    # TODO: do not compute multiple times the mean if called multiple times
    @property
    def vertexes_mean(self) -> np.ndarray:
        return self.vertexes.mean(0)

    # TODO: do not compute multiple times the mean if called multiple times
    @property
    def vertexes_std(self) -> np.ndarray:
        return self.vertexes.std(0)

    @property
    def centered_norm(self) -> float:
        return np.linalg.norm(self.vertexes - self.vertexes_mean)

    @property
    def Rotref(self) -> np.ndarray:
        assert self.Tref is not None
        return self.Tref[:3, :3] * np.sqrt(3) / np.linalg.norm(self.Tref[:3, :3])

    @property
    def sample(self) -> np.ndarray:
        return self.vertexes[self.sample_idx]

    def register_sample_to_reference(
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
        if self.sample_idx is None:
            warnings.warn("No sample found. Performing sampling.")
            self.perform_sampling(len(self.reference.sample_idx))

        T, indices, errs, n_iters = register_icp_hungarian(
            reference.sample, self.sample,
            allow_reflection=allow_reflection, init_pose=init_pose, max_iterations=max_iterations, tolerance=tolerance,)
        self.Tref = T
        self.sample_idx = self.sample_idx[indices]

        return self.Tref, self.sample_idx, errs, n_iters

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

        T, indices, errs, n_iters = register_icp(
            reference.vertexes, self.vertexes,
            allow_reflection=allow_reflection, init_pose=init_pose, max_iterations=max_iterations, tolerance=tolerance,)
        self.Tref = T
        # self.sample_idx = indices

        return self.Tref, indices, errs, n_iters

    def match_samples(self, reference: "Shape" = None, matching_method: str = "perfect") -> (np.ndarray, np.ndarray):
        """ Creates samples matches with reference's samples.
        """
        if reference is None:
            reference = self.reference

        if self.Tref is None:
            warnings.warn("No registration already done. Performing ICP registration with default parameters...")
            self.register_icp_to_reference(reference)

        if matching_method == "nearest":
            _, neigh_sampling = nearest_neighbor(
                transform_cloud(self.Tref, reference.sample),
                self.vertexes
            )
            self.sample_idx = np.array(neigh_sampling)

        elif matching_method in ["hungarian", "perfect", "perfect_matching"]:
            if self.sample_idx is None:
                warnings.warn("No sample found. Performing sampling.")
                self.perform_sampling(len(self.reference.sample_idx))
            neigh_sampling = perfect_matching(
                transform_cloud(self.Tref, reference.sample),
                self.sample
            )
            self.sample_idx = self.sample_idx[neigh_sampling]

        # self.sample = self.vertexes[neigh_sampling]
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
            self.sample_idx, self.dist_to_sample, self.closest_sample_point = dijkstra_sampling(
                verts=self.vertexes,
                faces=self.faces,
                n_points=n_points,
                verbose=verbose,
            )
        else:
            raise NotImplementedError

        self.sample_idx = np.array(self.sample_idx)
        return self.sample, self.dist_to_sample, self.closest_sample_point

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

    def plot_compare_point_cloud(self, ax, show_sampling=False, **kwargs):
        Tref_vert = transform_cloud(self.Tref, self.reference.vertexes)
        ax.scatter(*Tref_vert.T, label='ref', **kwargs)
        ax.scatter(*self.vertexes.T, label='shape', **kwargs)

        if show_sampling:
            Tsampling = transform_cloud(self.Tref, self.reference.sample)
            self.plot_link_samples(ax, Tsampling, self.sample)

        ax.legend()
        return ax

    def plot_compare_samples(self, ax, **kwargs):
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
        for s1, s2 in zip(samples1, samples2):
            toplot = [[s1[i], s2[i]] for i in range(3)]
            ax.plot(*toplot, **kwargs)
        return ax

    def apply_procrustes(self, reference):
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

    def dijkstra_to_sample(self) -> (np.ndarray, np.ndarray):
        self.dist_to_sample, self.closest_sample_point = dijkstra_mesh(self.vertexes, self.faces, self.sample_idx)
        return self.dist_to_sample, self.closest_sample_point

    def compute_sample_faces(self) -> np.ndarray:
        """ Uses voronoi graph to connect edges. If two voronoi cells touch, the points are connected.
        """
        assert self.sample_idx is not None
        gsample = nx.Graph()
        sample_order = {self.sample_idx[k]: k for k in range(len(self.sample_idx))}

        gmesh = create_mesh_graph(self.vertexes, self.faces)
        visited_edges = set()
        for node1 in gmesh.nodes:
            for node2 in gmesh.edges[node1]:
                if (node2, node1) in visited_edges:
                    continue
                visited_edges.add((node1, node2))
                c1 = self.closest_sample_point[node1]
                c2 = self.closest_sample_point[node2]
                if c1 != c2:
                    gsample.add_edge(sample_order[c1], sample_order[c2])

        self.faces_sample = np.array([clique for clique in nx.enumerate_all_cliques(gsample) if len(clique) == 3])

        return self.faces_sample

    def o3d_mesh(self, vertex_colors="dist_to_sample") -> "open3d.cpu.pybind.geometry.TriangleMesh":

        kwargs = {'vertices': self.vertexes, 'triangles': self.faces}
        if vertex_colors == 'dist_to_sample' and self.dist_to_sample is not None:
            kwargs['vertex_colors'] = colormap_1d(max_min_norm(self.dist_to_sample))

        cur_mesh = numpy_to_o3d_mesh(**kwargs)
        cur_mesh.compute_vertex_normals()

        return cur_mesh

    def o3d_ref_mesh_transformed(self, vertex_colors="dist_to_sample") -> "open3d.cpu.pybind.geometry.TriangleMesh":
        kwargs = {
            'vertices': transform_cloud(self.Tref, self.reference.vertexes),
            'triangles': self.reference.faces,
        }
        if vertex_colors == 'dist_to_sample' and self.dist_to_sample is not None:
            kwargs['vertex_colors'] = colormap_1d(max_min_norm(self.reference.dist_to_sample))

        ref_mesh = numpy_to_o3d_mesh(**kwargs)
        ref_mesh.compute_vertex_normals()

        return ref_mesh

    def o3d_pcd(self, point_color='g', only_sample=True, **kwargs) -> "open3d.cpu.pybind.geometry.PointCloud":
        to_plot = self.vertexes
        if only_sample and self.sample_idx is not None:
            to_plot = self.sample

        return get_o3d_pcd_colored(to_plot, point_color, **kwargs)

    def o3d_ref_pcd_transformed(self, point_color='g', only_sample=True, **kwargs) -> "open3d.cpu.pybind.geometry.PointCloud":
        assert self.Tref is not None
        to_plot = transform_cloud(self.Tref, self.reference.vertexes)
        if only_sample and self.sample_idx is not None:
            to_plot = transform_cloud(self.Tref, self.reference.sample)

        return get_o3d_pcd_colored(to_plot, point_color, **kwargs)

    def create_mesh_from_sample_faces(self, points: np.ndarray, **kwargs) -> "open3d.cpu.pybind.geometry.TriangleMesh":
        if self.faces_sample is None:
            warnings.warn("Faces sample not yet computed. Is computed.")
            self.compute_sample_faces()
        mesh = numpy_to_o3d_mesh(
            vertices=points,
            triangles=self.faces_sample,
            **kwargs
        )
        mesh.compute_vertex_normals()
        return mesh


class SSM:

    def __init__(self, shapes: List[Shape] = None, reference: Shape = None):
        self.shapes = shapes
        self.reference = reference

        self.pca: PCA = None
        self.pca_basis: List[Shape] = None
        self.all_samples_mean: np.ndarray = None
        self.inliers: np.ndarray = None  # opposed to outliers

    @staticmethod
    def detect_outliers_sample(shape, **kwargs):
        all_samples = shape.all_samples.reshape(len(shape), -1)
        diff_mean = np.zeros(len(shape))

        for idx in range(len(shape)):
            diff_mean[idx] = np.abs(all_samples[idx] - array_wo_idx(all_samples, idx).mean()).mean()

        return ~is_outlier_1d(diff_mean, **kwargs)

    def compute_pca(self, remove_outliers=True, **outlier_removal_args):
        if remove_outliers:
            self.inliers = SSM.detect_outliers_sample(self, **outlier_removal_args)
        else:
            self.inliers = np.ones(len(self)).astype(bool)
        all_samples_inline = self.all_samples.reshape(len(self), -1)[self.inliers]
        self.pca = PCA(n_components=min(len(all_samples_inline), np.prod(self.size)))
        self.all_samples_mean = all_samples_inline.mean(0)
        self.pca.fit(all_samples_inline - self.all_samples_mean)

    @property
    def all_samples(self) -> np.ndarray:
        return np.stack(
            [transform_cloud(np.linalg.inv(shape.Tref), shape.sample) for shape in self.shapes],
            axis=0
        )

    @property
    def all_sample_normals(self) -> np.ndarray:
        """ Applies the inverse rotation reference to the normals of the shapes.
        If R @ ref.T -> Shape.T, then ref @ R.T -> Shape, then Shape @ R -> ref
        """
        return np.stack(
            [shape.normals[shape.sample_idx] @ shape.Rotref for shape in self.shapes],
            axis=0
        )

    def __len__(self):
        return len(self.shapes)

    def get_component(self, idx: int) -> np.ndarray:
        return (self.pca.components_[idx] * 3 * np.sqrt(self.pca.explained_variance_[idx]) +
                self.all_samples_mean).reshape(*self.size)

    def in_pca_basis(self, coords: np.ndarray) -> np.ndarray:
        return (coords @ self.pca.components_[:len(coords)] + self.all_samples_mean).reshape(*self.size)

    @property
    def size(self) -> Tuple:
        return self.shapes[0].sample.shape

    def random_pca_features(self, scalar: float = 3, n_pca: int = None,
            lb: np.ndarray = None, ub: np.ndarray = None) -> np.ndarray:
        """
        Returns a random generated shape from the principal components.
        Args:
            scalar (float): how to multiply the eigenvalue for the generation
            n_pca (int): number of principal components to take
            lb (np.ndarray): shape (n_pca). Lower bound for sampling.
            ub (np.ndarray): shape (n_pca). Upper bound for sampling.

        Returns:
            np.ndarray: shape (n_pca. feature vector.
        """
        if n_pca is None:
            n_pca = len(self.pca.explained_variance_)

        if ub is None:
            ub = scalar * np.sqrt(self.pca.explained_variance_)
        if lb is None:
            lb = -scalar * np.sqrt(self.pca.explained_variance_)
        return uniform_sampling_bound(lb[:n_pca], ub[:n_pca])

    def random_sampling_pca(self, **random_kwargs):
        """
        Returns a random generated shape from the principal components.
        Args:
            see random_pca_features args
        Returns:
            (np.ndarray, np.ndarray): shape (N_points, 3), shape (n_pca). The generated point cloud and the feature
                                      vector.
        """
        features = self.random_pca_features(**random_kwargs)
        return self.in_pca_basis(features), features

    def random_pcd_pca(self, point_color: str = 'g', kwargs_pcd: Dict = {}, **random_kwargs):
        features = self.random_pca_features(**random_kwargs)
        return self.in_pca_basis_pcd(features, point_color, **kwargs_pcd), features

    def random_mesh_pca(self, point_color: str = 'g', shape_reference_faces: Shape = None, kwargs_mesh: Dict = {}, kwargs_pcd: Dict = {}, **random_kwargs):
        features = self.random_pca_features(**random_kwargs)
        return self.in_pca_basis_mesh(features, shape_reference_faces, point_color, kwargs_pcd=kwargs_pcd, **kwargs_mesh), features

    def in_pca_basis_pcd(self, b: np.ndarray, point_color: str = 'g', **kwargs) -> "open3d.cpu.pybind.geometry.PointCloud":
        return get_o3d_pcd_colored(self.in_pca_basis(b), point_color, **kwargs)

    def in_pca_basis_mesh(
        self,
        b: np.ndarray,
        shape_reference_faces: Shape = None,
        point_color: str = 'g',
        kwargs_pcd: Dict = {},
        **kwargs
    ) -> ("open3d.cpu.pybind.geometry.TriangleMesh", "open3d.cpu.pybind.geometry.PointCloud"):
        pcd = self.in_pca_basis_pcd(b, point_color, **kwargs_pcd)
        if shape_reference_faces is None:
            shape_reference_faces = random.choice(self.shapes)
        return shape_reference_faces.create_mesh_from_sample_faces(np.asarray(pcd.points)), pcd
