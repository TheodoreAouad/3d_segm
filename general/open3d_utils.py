import open3d as o3d


def numpy_to_o3d_pcd(points: "np.ndarray", **kwargs) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    for attr, value in kwargs.items():
        type_fn = o3d.utility.Vector3dVector
        setattr(pcd, attr, type_fn(value))

    return pcd


def numpy_to_o3d_mesh(**kwargs) -> o3d.geometry.TriangleMesh:
    msh = o3d.geometry.TriangleMesh()
    # msh.vertices = o3d.utility.Vector3dVector(verts)
    # msh.triangles = o3d.utility.Vector3iVector(faces)
    # if vertex_normals is not None:
    #     msh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
    for attr, value in kwargs.items():
        if attr == "triangles":
            type_fn = o3d.utility.Vector3iVector
        else:
            type_fn = o3d.utility.Vector3dVector
        setattr(msh, attr, type_fn(value))
    return msh
