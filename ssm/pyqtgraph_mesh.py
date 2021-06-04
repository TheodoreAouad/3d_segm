from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import sys


# Load data

def plot_mesh(
    vertexes,
    faces,
    color_vertexes=None,
    color_faces=None,
    title='',
    points=[],
    size_point=20,
    point_color=[0,1,0,1],
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
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.show()
    w.setWindowTitle(title)

    mesh = gl.GLMeshItem(
        vertexes=vertexes,
        faces=faces,
        faceColors=color_faces,
        vertexColors=color_vertexes,
    )

    w.addItem(mesh)

    for coords in points:
        mesh = gl.MeshData.sphere(size_point, size_point)
        colors = np.array([point_color for _ in range(len(mesh.vertexes()))])
        sphere = gl.GLMeshItem(
            vertexes=mesh.vertexes() + coords,
            faces=mesh.faces(),
            vertexColors=colors,
        )
        w.addItem(sphere)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
