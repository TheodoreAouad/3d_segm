from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import sys


# Load data

def plot_mesh(
    vertexes=None,
    faces=None,
    color_vertexes=None,
    color_faces=None,
    title='',
    points=[],
    size_point=20,
    point_color=[0, 1, 0, 1],
):
    """
    Plot the interactive pyqtgraph for a mesh.
    Args:
        vertexes (np.ndarray): (N x d), N vertexes of dimension d
        faces (np.ndarray): (M x 3), triangles of int containing the triangles (the idxes of vertexes)
        color_vertexes (np.ndarray): (N x 4) RGBA value for each vertex
        color_faces (np.ndarray): (M x 4) RGBA value for each face
        title (str): title of the mesh graph
        points (np.ndarray): (K x d) coordinates of points to plot as sphere
        size_point (float): size of a point
        point_color (array-like): len = 4: RGBA of the point

    """
    plotter = PlotMesh(title="SSM")
    plotter.add_mesh(vertexes, faces, color_faces=color_faces, color_vertexes=color_vertexes)
    plotter.add_point_cloud(points, size_point=size_point, point_color=point_color)
    plotter.show()


class PlotMesh:

    def __init__(self, title=''):
        """
        Creates the widget and shows it.

        Args:
            self: write your description
            title: write your description
        """
        self.app = QtGui.QApplication.instance()
        if self.app is None:
            self.app = QtGui.QApplication([])
        self.w = gl.GLViewWidget()
        self.w.show()
        self.w.setWindowTitle(title)

        self.meshes = []
        self.point_clouds = []

    def add_mesh(self, *args, **kwargs):
        """
        Add a mesh item to the scene.

        Args:
            self: write your description
        """
        mesh = gl.GLMeshItem(*args, **kwargs)

        self.w.addItem(mesh)
        self.meshes.append(mesh)

    def add_point_cloud(self, points, size_point=20, point_color=[0, 1, 0, 1]):
        """
        Add a point cloud to the plot.

        Args:
            self: write your description
            points: write your description
            size_point: write your description
            point_color: write your description
        """
        for coords in points:
            mesh = gl.MeshData.sphere(size_point, size_point)
            colors = np.array([point_color for _ in range(len(mesh.vertexes()))])
            sphere = gl.GLMeshItem(
                vertexes=mesh.vertexes() + coords,
                faces=mesh.faces(),
                vertexColors=colors,
            )
            self.w.addItem(sphere)

        self.point_clouds.append(points)

    def show(self):
        """
        Show the main window.

        Args:
            self: write your description
        """
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def centroid_mesh(self, idx):
        """
        Return the centroid of the mesh at the given index.

        Args:
            self: write your description
            idx: write your description
        """
        return self.meshes[idx].vertexes.mean(0)

    def centroid_point_cloud(self, idx):
        """
        Point cloud of the centroid.

        Args:
            self: write your description
            idx: write your description
        """
        return self.point_clouds[idx].vertexes.mean(0)

    def centroid_all_meshes(self, idx):
        """
        Calculate the centroid of all meshes.

        Args:
            self: write your description
            idx: write your description
        """
        centroid = np.zeros((len(self.meshes), 3))
        for idx in range(len(self.meshes)):
            centroid[idx] = self.centroid_mesh(idx)
        return centroid.mean(0)
