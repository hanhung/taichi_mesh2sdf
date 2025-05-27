import trimesh
import numpy as np
import taichi as ti
from tqdm import tqdm

from ti_mesh2sdf import utils

def compute_watertight_mesh(obj_file, resolution, exact_band):
    level = 2 / resolution
    vertices, faces = utils.get_norm_vert_face(obj_file, mesh_scale=0.9)
    bbmin, bbmax, res_xyz, d_xyz = utils.get_grid_params(resolution, vertices, margin=0.1)

    # Init sdf grid and intersection count fields
    phi = ti.field(ti.f32, shape=res_xyz)
    phi.fill(res_xyz.sum() * d_xyz)
    intersection_count = ti.field(ti.i32, shape=res_xyz)
    intersection_count.fill(0)

    # Create triangle and vertices in taichi fields
    tris = ti.field(ti.i32, shape=faces.shape)
    tris.from_numpy(faces)
    verts = ti.field(ti.f32, shape=vertices.shape)
    verts.from_numpy(vertices)

    origin = utils.vec3([bbmin[0], bbmin[1], bbmin[2]])
    ti_res_xyz = utils.vec3i(res_xyz)

    # Calculate UDF for input mesh
    utils.compute_udf(phi, intersection_count, tris, verts, origin, ti_res_xyz, d_xyz, exact_band)

    watertight_mesh, _, _ = utils.get_watertight_norm_mesh(phi, level, resolution)
    return watertight_mesh

def compute_sdf_from_watertight(watertight_mesh, resolution, exact_band, mesh_scale=0.9):
    if type(watertight_mesh) == str:
        vertices, faces = utils.get_norm_vert_face(watertight_mesh, mesh_scale=mesh_scale)
    else:
        vertices = watertight_mesh.vertices.astype(np.float32)
        faces = watertight_mesh.faces.astype(np.int32)
    bbmin, bbmax, res_xyz, d_xyz = utils.get_grid_params(resolution, vertices, margin=0.1)
    
    # Init sdf grid and intersection count fields
    phi = ti.field(ti.f32, shape=res_xyz)
    phi.fill(res_xyz.sum() * d_xyz)
    intersection_count = ti.field(ti.i32, shape=res_xyz)
    intersection_count.fill(0)

    # Re-create triangle and vertices in taichi fields
    tris = ti.field(ti.i32, shape=faces.shape)
    tris.from_numpy(faces)
    verts = ti.field(ti.f32, shape=vertices.shape)
    verts.from_numpy(vertices)

    origin = utils.vec3([bbmin[0], bbmin[1], bbmin[2]])
    ti_res_xyz = utils.vec3i(res_xyz)

    # Calculate SDF for watertight mesh
    utils.compute_udf(phi, intersection_count, tris, verts, origin, ti_res_xyz, d_xyz, exact_band)
    utils.compute_sign(phi, intersection_count, ti_res_xyz)

    return phi.to_numpy()

def compute_occ_flood_fill(occ):
    occupancy = ti.field(dtype=ti.i32, shape=occ.shape)
    occupancy.from_numpy(occ)
    visited = ti.field(dtype=ti.i32, shape=occ.shape)
    visited.fill(0)

    res_xyz = utils.vec3i(occ.shape)
    
    utils.initialize_frontier(occupancy, visited, res_xyz)  # Initialize the boundary voxels
    for _ in tqdm(range(max(occupancy.shape) * 2)):  # Conservative upper limit on iterations
        utils.flood_fill_pass(occupancy, visited, res_xyz)  # Expand outward from the boundary
    utils.label_interior(occupancy, visited, res_xyz)  # Mark unvisited regions as interior

    return occupancy.to_numpy()
