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
mesh_path = all_meshes[4]
print("Reference:", ref_mesh)
ref_verts = np.load(os.path.join(ref_mesh, "vertexes.npy"))
ref_faces = np.load(os.path.join(ref_mesh, "faces.npy"))

ref_shape = sh.Shape(label=ref_mesh, vertexes=ref_verts, faces=ref_faces)
ref_shape.perform_sampling(n_samples)


verts = np.load(os.path.join(mesh_path, 'vertexes.npy'))
faces = np.load(os.path.join(mesh_path, 'faces.npy'))

cur_shape = sh.Shape(label=mesh_path, vertexes=verts, faces=faces)
cur_shape.set_reference(ref_shape)

# %%

_, errs, n_iters = cur_shape.register_icp_to_reference()
cur_shape.match_samples()


# %%
reload_modules()
plotter = pm.PlotMesh(title='SSM')
Tref_verts = su.transform_cloud(cur_shape.Tref, ref_shape.vertexes)

plotter.add_point_cloud(Tref_verts - cur_shape.vertexes.mean(0), point_color=[0, 1, 0, 1], size_point=5)
plotter.add_point_cloud(cur_shape.vertexes - cur_shape.vertexes.mean(0), point_color=[1, 0, 0, 1], size_point=5)


plotter.show()