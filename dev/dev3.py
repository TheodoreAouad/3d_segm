from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import itertools

import os
import nibabel as nib
import general.utils as u
import skimage.measure as meas
from tqdm import tqdm

import ssm.dijkstra as dij


N_POINTS = 50
PATH_SEGM = os.path.abspath("/hdd/datasets/CT-ORG/raw/labels_and_README/labels-11.nii.gz")


# Load data
print('loading image...')
seg1n = nib.load(PATH_SEGM)
seg1 = np.round(seg1n.get_fdata()) == 2
reg1 = (u.get_most_important_regions(seg1) > 0).astype(int)
verts1, faces1, normals1, values1 = meas.marching_cubes(reg1, step_size=3)
print('Done...')

# compute distances
gmesh = dij.Graph()
for tri in faces1:
    for i in range(2):
        for j in range(i + 1, 3):
            gmesh.add_node(tri[i])
            gmesh.add_node(tri[j])
            value = np.linalg.norm(verts1[tri[i]] - verts1[tri[j]])
            gmesh.add_edge(tri[i], tri[j], value)
            gmesh.add_edge(tri[j], tri[i], value)

cur_point = 0

all_points = []
for _ in tqdm(range(N_POINTS)):
    all_points.append(cur_point)
    visited, path = dij.dijsktra(gmesh, initial_set=all_points)
    ar_dist = np.zeros(len(gmesh.nodes))
    for key, value in visited.items():
        ar_dist[key] = value
    cur_point = ar_dist.argmax()

colors_faces = (ar_dist[faces1].mean(1) - ar_dist[faces1].min()) / (ar_dist[faces1].max() - ar_dist[faces1].min())
colors_faces = np.stack(
    (colors_faces, np.zeros_like(colors_faces), 1 - colors_faces, np.ones_like(colors_faces)), axis=-1
)
color_vertices = (ar_dist - ar_dist.min()) / (ar_dist.max() - ar_dist.min())
color_vertices = np.stack(
    (color_vertices, np.zeros_like(color_vertices), 1 - color_vertices, np.ones_like(color_vertices)), axis=-1
)

app = QtGui.QApplication.instance()
if app is None:
    app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('A cube')

verts1_mean = verts1.mean(0)

cube = gl.GLMeshItem(
    vertexes=verts1 - verts1_mean,
    faces=faces1,
    # faceColors=colors_faces,
    vertexColors=color_vertices,
)

w.addItem(cube)

for point in all_points:
    coords = verts1[point]
    mesh = gl.MeshData.sphere(20, 20)
    colors = np.array([[0, 1, 0, 1] for _ in range(len(mesh.vertexes()))])
    sphere = gl.GLMeshItem(
        vertexes=mesh.vertexes() + coords - verts1_mean,
        faces=mesh.faces(),
        # faceColors=colors_faces,
        vertexColors=colors,
    )
    w.addItem(sphere)


if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
