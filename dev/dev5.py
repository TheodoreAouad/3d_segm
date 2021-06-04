from ssm.pyqtgraph_mesh import plot_mesh

import os
import nibabel as nib
import general.utils as u
import numpy as np
import skimage.measure as meas
from tqdm import tqdm

import ssm.dijkstra as dij

# Load data
print('loading image...')
path_segm = os.path.abspath("/hdd/datasets/CT-ORG/raw/labels_and_README/labels-11.nii.gz")
seg1n = nib.load(path_segm)
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

n_points = 4
cur_point = 0

all_points = []
for _ in tqdm(range(n_points)):
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

verts1 = verts1 - verts1.mean(0)
plot_mesh(verts1, faces1, color_vertices, points=verts1[all_points])

