from importlib import reload
import os

import numpy as np
from tqdm import tqdm

import ssm.utils as su
import ssm.shape as sh
import ssm.pyqtgraph_mesh as pm

def reload_modules():
    for modl in [pm]:
        reload(pm)

# %%
# Using shape class

mesh_dir = "/hdd/datasets/CT-ORG/meshes/labels/"
step_size = 4
n_samples = 20

all_meshes = su.sort_by_regex([
    os.path.join(mesh_dir, filename, f"step_size_{step_size}") for filename in os.listdir(mesh_dir)
])
ref_mesh = all_meshes[2]
print("Reference:", ref_mesh)
ref_verts = np.load(os.path.join(ref_mesh, "vertexes.npy"))
ref_faces = np.load(os.path.join(ref_mesh, "faces.npy"))

ref_shape = sh.Shape(label=ref_mesh, vertexes=ref_verts, faces=ref_faces)
ref_shape.perform_sampling(n_samples)

# %%

n_shapes = 10
all_shapes = []
all_errs = []
all_n_iters = []

iterator = su.sort_by_regex(list(set(all_meshes).difference([ref_mesh])))[:n_shapes]
for idx in tqdm(range(len(iterator))):
    mesh_path = iterator[idx]
    verts = np.load(os.path.join(mesh_path, 'vertexes.npy'))
    faces = np.load(os.path.join(mesh_path, 'faces.npy'))
    cur_shape = sh.Shape(label=mesh_path, vertexes=verts, faces=faces)
    cur_shape.set_reference(ref_shape)
    _, errs, n_iters = cur_shape.register_icp_to_reference()
    cur_shape.match_samples()

    all_shapes.append(cur_shape)
    all_errs.append(errs)
    all_n_iters.append(n_iters)


# %%

shape_to_plots = [all_shapes[4], all_shapes[5]]
color_vertices = []
verts_mean = []


for idx in range(len(shape_to_plots)):
    shape_to_plots[idx].dijkstra_to_sample()
    color_vertices.append(
        (shape_to_plots[idx].dist_to_sample - shape_to_plots[idx].dist_to_sample.min()) / (shape_to_plots[idx].dist_to_sample.max() - shape_to_plots[idx].dist_to_sample.min())
    )
    color_vertices[0] = np.stack(
        (color_vertices[0], np.zeros_like(color_vertices[0]), 1 - color_vertices[0], np.ones_like(color_vertices[0])), axis=-1
)
verts_mean.append(shape_to_plots[0].vertexes.mean(0))


shape_to_plots[1].dijkstra_to_sample()
color_vertices.append((shape_to_plots[1].dist_to_sample - shape_to_plots[1].dist_to_sample.min()) / (shape_to_plots[1].dist_to_sample.max() - shape_to_plots[1].dist_to_sample.min()))
color_vertices[1] = np.stack(
    (color_vertices[1], np.zeros_like(color_vertices[1]), 1 - color_vertices[1], np.ones_like(color_vertices[1])), axis=-1
)
verts_mean.append(shape_to_plots[1].vertexes.mean(0))


# %%
reload_modules()
# pm.plot_mesh(shape_to_plot.vertexes - verts_mean, shape_to_plot.faces, color_vertices, points=shape_to_plot.sample - verts_mean)
plotter = pm.PlotMesh(title='SSM')
plotter.add_mesh(shape_to_plots[0].vertexes - verts_mean[0], shape_to_plots[0].faces, color_vertexes=color_vertices[0])
plotter.add_point_cloud(shape_to_plots[0].sample - verts_mean[0])

plotter.add_mesh(shape_to_plots[1].vertexes - verts_mean[1] + 100 * np.ones(3), shape_to_plots[1].faces, color_vertexes=color_vertices[1])
plotter.add_point_cloud(shape_to_plots[1].sample - verts_mean[1]  + 100 * np.ones(3))


plotter.show()