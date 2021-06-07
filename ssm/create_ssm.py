import os
import random

import numpy as np
import tqdm as tqdm
import matplotlib.pyplot as plt

from .utils import sort_by_regex
from .shape import Shape

mesh_dir = "/hdd/datasets/CT-ORG/meshes/labels/"
step_size = 4

all_meshes = sort_by_regex([
    os.path.join(mesh_dir, filename, f"step_size_{step_size}") for filename in os.listdir(mesh_dir)
])
ref_mesh = all_meshes[2]
print(ref_mesh)
ref_verts = np.load(os.path.join(ref_mesh, "vertexes.npy"))
ref_faces = np.load(os.path.join(ref_mesh, "faces.npy"))

ref_shape = Shape(label=ref_mesh, vertexes=ref_verts, faces=ref_faces)
ref_shape.perform_sampling(20)


n_shapes = 10
all_shapes = []
all_errs = []
all_n_iters = []

iterator = sort_by_regex(list(set(all_meshes).difference([ref_mesh])))[:n_shapes]
for idx in tqdm(range(len(iterator))):
    mesh_path = iterator[idx]
    verts = np.load(os.path.join(mesh_path, 'vertexes.npy'))
    cur_shape = Shape(label=mesh_path, vertexes=verts)
    cur_shape.set_reference(ref_shape)
    _, errs, n_iters = cur_shape.register_icp_to_reference()
    cur_shape.align_samples()

    all_shapes.append(cur_shape)
    all_errs.append(errs)
    all_n_iters.append(n_iters)


cur_shape = random.choice(all_shapes)
print(cur_shape.label)
print(cur_shape.Tref)

fig = plt.figure(figsize=(7, 7))
ax1 = fig.add_subplot(121, projection='3d')
cur_shape.plot_compare_point_cloud(ax1)

ax1 = fig.add_subplot(122, projection='3d')
cur_shape.plot_compare_samples(ax1)
