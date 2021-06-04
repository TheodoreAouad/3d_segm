import pyqtgraph.opengl as gl
mesh = gl.MeshData.sphere(20, 20)

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import itertools


app = QtGui.QApplication.instance()
if app is None:
    app = QtGui.QApplication([])
w = gl.GLViewWidget()
# w.opts['distance'] = 20
w.show()
w.setWindowTitle('A cube')

cube = gl.GLMeshItem(
    vertexes=mesh.vertexes(),
    faces=mesh.faces(),
    # faceColors=colors_faces,
    # vertexColors=color_vertices,
)

w.addItem(cube)

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

import pyqtgraph.opengl as gl
mesh = gl.MeshData.sphere(5, 2)