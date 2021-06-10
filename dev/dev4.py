import os

import numpy as np
from tqdm import tqdm

import ssm.utils as su
import ssm.shape as sh
from ssm.pyqtgraph_mesh import plot_mesh

# Using shape class

mesh_dir = "/hdd/datasets/CT-ORG/meshes/labels/"
step_size = 4
n_samples = 100

all_meshes = su.sort_by_regex([
    os.path.join(mesh_dir, filename, f"step_size_{step_size}") for filename in os.listdir(mesh_dir)
])
ref_mesh = all_meshes[2]
print("Reference:", ref_mesh)
ref_verts = np.load(os.path.join(ref_mesh, "vertexes.npy"))
ref_faces = np.load(os.path.join(ref_mesh, "faces.npy"))

ref_shape = sh.Shape(label=ref_mesh, vertexes=ref_verts, faces=ref_faces)
ref_shape.perform_sampling(n_samples)


n_shapes = 10
all_shapes = []
all_errs = []
all_n_iters = []

iterator = su.sort_by_regex(list(set(all_meshes).difference([ref_mesh])))[:n_shapes]
for idx in tqdm(range(len(iterator))):
    mesh_path = iterator[idx]
    verts = np.load(os.path.join(mesh_path, 'vertexes.npy'))
    cur_shape = sh.Shape(label=mesh_path, vertexes=verts)
    cur_shape.set_reference(ref_shape)
    _, errs, n_iters = cur_shape.register_icp_to_reference()
    cur_shape.match_samples()

    all_shapes.append(cur_shape)
    all_errs.append(errs)
    all_n_iters.append(n_iters)




shape_to_plot =all_shapes[7]

color_vertices = (shape_to_plot.dist_to_sample - shape_to_plot.dist_to_sample.min()) / (shape_to_plot.dist_to_sample.max() - shape_to_plot.dist_to_sample.min())
color_vertices = np.stack(
    (color_vertices, np.zeros_like(color_vertices), 1 - color_vertices, np.ones_like(color_vertices)), axis=-1
)
verts_mean = shape_to_plot.vertexes.mean(0)
plot_mesh(shape_to_plot.vertexes - verts_mean, shape_to_plot.faces, color_vertices, points=shape_to_plot.sample - verts_mean)
