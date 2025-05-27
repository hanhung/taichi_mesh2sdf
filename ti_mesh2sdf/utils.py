import trimesh
import numpy as np
import taichi as ti
import skimage.measure

vec3 = ti.types.vector(3, ti.f32)
vec3i = ti.types.vector(3, ti.i32)

def get_norm_vert_face(obj_path, mesh_scale=0.9):
    if type(obj_path) == str:
        mesh = trimesh.load(obj_path, force='mesh')
    else:
        mesh = obj_path

    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)

    return vertices, faces

def get_watertight_norm_mesh(phi, level, resolution, mesh_scale=0.9):
    sdf = phi.to_numpy()
    sdf = np.abs(sdf)
    vertices, faces, _, _ = skimage.measure.marching_cubes(sdf, level)

    mesh = trimesh.Trimesh(vertices, faces)
    mesh.vertices = mesh.vertices * (2.0 / resolution) * mesh_scale - 1.0
    return mesh, mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)

def get_grid_params(base_resolution, vertices, margin=0.1):
    # Get base resolution and calculate bbox of object (add margins)
    bbmin = vertices.min(0) - margin
    bbmax = vertices.max(0) + margin

    # Get the voxel resolutions depending on bbox sizes, should have about the same distance between voxels (d_xyz)
    res_xyz = np.array([base_resolution * (bbmax[0] - bbmin[0]) / 2, 
                        base_resolution * (bbmax[1] - bbmin[1]) / 2,
                        base_resolution * (bbmax[2] - bbmin[2]) / 2], dtype=np.int32)
    d_xyz = (bbmax[0] - bbmin[0]) / res_xyz[0]

    return bbmin, bbmax, res_xyz, d_xyz

@ti.func
def point_segment_distance(x0: vec3,
                           x1: vec3,
                           x2: vec3) -> ti.f32:
    dx = x2 - x1
    m2 = dx.norm_sqr()
    s12 = (x2 - x0).dot(dx) / m2

    s12 = ti.max(0.0, ti.min(1.0, s12))
    closest_point = s12 * x1 + (1.0 - s12) * x2
    return (x0 - closest_point).norm()

@ti.func
def point_triangle_distance(x0: vec3,
                            x1: vec3,
                            x2: vec3,
                            x3: vec3) -> ti.f32:
    # Compute vectors for the triangle edges
    x13 = x1 - x3
    x23 = x2 - x3
    x03 = x0 - x3

    m13 = x13.norm_sqr()  # Squared magnitude of x13
    m23 = x23.norm_sqr()  # Squared magnitude of x23
    d = x13.dot(x23)  # Dot product of x13 and x23

    # Compute the inverse determinant for barycentric coordinates
    invdet = 1.0 / ti.max(m13 * m23 - d * d, 1e-30)
    a = x13.dot(x03)
    b = x23.dot(x03)

    # Barycentric coordinates of the closest point on the triangle's plane
    w23 = invdet * (m23 * a - d * b)
    w31 = invdet * (m13 * b - d * a)
    w12 = 1.0 - w23 - w31

    # Initialize the distance variable to a default large value
    distance = 1e10  # Arbitrary large value

    # Check if the point is inside the triangle
    if w23 >= 0.0 and w31 >= 0.0 and w12 >= 0.0:
        closest_point = w23 * x1 + w31 * x2 + w12 * x3
        distance = (x0 - closest_point).norm()
    else:
        dist1 = point_segment_distance(x0, x1, x2)
        dist2 = point_segment_distance(x0, x1, x3)
        dist3 = point_segment_distance(x0, x2, x3)

        if w23 > 0.0:  # This rules out edge 2-3
            distance = ti.min(dist1, dist2)
        elif w31 > 0.0:  # This rules out edge 1-3
            distance = ti.min(dist1, dist3)
        else:  # w12 > 0.0, rules out edge 1-2
            distance = ti.min(dist2, dist3)

    return distance

@ti.func
def orientation(x1: ti.f32, 
                y1: ti.f32, 
                x2: ti.f32, 
                y2: ti.f32, 
                twice_signed_area: ti.template()) -> int:
    """
    Calculates the twice signed area of the triangle (0, 0) -> (x1, y1) -> (x2, y2).
    Returns -1, 0, or 1 based on the SOS-determined orientation.
    """
    results = 0

    twice_signed_area[0] = y1 * x2 - x1 * y2  # Compute cross product for the area
    if twice_signed_area[0] > 0.0:
        results = 1
    elif twice_signed_area[0] < 0.0:
        results = -1
    elif y2 > y1:
        results = 1
    elif y2 < y1:
        results = -1
    elif x1 > x2:
        results = 1
    elif x1 < x2:
        results = -1
    else:
        results = 0  # Degenerate case
    return results

@ti.func
def point_in_triangle_2d(x0: ti.f32, y0: ti.f32,
                         x1: ti.f32, y1: ti.f32,
                         x2: ti.f32, y2: ti.f32,
                         x3: ti.f32, y3: ti.f32,
                         a: ti.template(), b: ti.template(), c: ti.template()) -> bool:
    """
    Robust test for whether (x0, y0) is inside the triangle (x1, y1)-(x2, y2)-(x3, y3).
    If true, the barycentric coordinates (a, b, c) are set.
    """
    # Translate points to set (x0, y0) as the origin
    x1, y1 = x1 - x0, y1 - y0
    x2, y2 = x2 - x0, y2 - y0
    x3, y3 = x3 - x0, y3 - y0

    results = True

    # Orientation tests
    signa = orientation(x2, y2, x3, y3, a)
    if signa == 0:
        results = False

    signb = orientation(x3, y3, x1, y1, b)
    if signb != signa:
        results = False

    signc = orientation(x1, y1, x2, y2, c)
    if signc != signa:
        results = False

    # Normalize the barycentric coordinates
    sum_abc = a[0] + b[0] + c[0]
    assert sum_abc != 0.0, "Sum of barycentric coordinates should not be zero"
    a[0] /= sum_abc
    b[0] /= sum_abc
    c[0] /= sum_abc

    return results

@ti.kernel
def compute_udf(phi_local: ti.template(),
                intersection_count_local: ti.template(),
                tris_local: ti.template(),
                verts_local: ti.template(),
                origin: vec3,
                res_xyz: vec3i,
                d_xyz: ti.f32,
                exact_band: ti.i32):
    for t in range(tris_local.shape[0]):
        # Triangle vertex indices
        p, q, r = tris_local[t, 0], tris_local[t, 1], tris_local[t, 2]

        verts_p = vec3((verts_local[p, 0], verts_local[p, 1], verts_local[p, 2]))
        verts_q = vec3((verts_local[q, 0], verts_local[q, 1], verts_local[q, 2]))
        verts_r = vec3((verts_local[r, 0], verts_local[r, 1], verts_local[r, 2]))

        # Grid coordinates for each vertex
        fip, fjp, fkp = (verts_p - origin) / d_xyz
        fiq, fjq, fkq = (verts_q - origin) / d_xyz
        fir, fjr, fkr = (verts_r - origin) / d_xyz

        # Bounds for the grid cells
        i0 = ti.max(0, ti.min(int(ti.floor(min(fip, fiq, fir))) - exact_band, res_xyz[0] - 1))
        i1 = ti.min(res_xyz[0] - 1, ti.max(int(ti.ceil(max(fip, fiq, fir))) + exact_band, 0))
        j0 = ti.max(0, ti.min(int(ti.floor(min(fjp, fjq, fjr))) - exact_band, res_xyz[1] - 1))
        j1 = ti.min(res_xyz[1] - 1, ti.max(int(ti.ceil(max(fjp, fjq, fjr))) + exact_band, 0))
        k0 = ti.max(0, ti.min(int(ti.floor(min(fkp, fkq, fkr))) - exact_band, res_xyz[2] - 1))
        k1 = ti.min(res_xyz[2] - 1, ti.max(int(ti.ceil(max(fkp, fkq, fkr))) + exact_band, 0))

        # Iterate over grid cells within the bounds
        for k in range(k0, k1 + 1):
            for j in range(j0, j1 + 1):
                for i in range(i0, i1 + 1):
                    gx = vec3([i * d_xyz + origin[0], j * d_xyz + origin[1], k * d_xyz + origin[2]])
                    d = point_triangle_distance(gx, verts_p, verts_q, verts_r)
                    ti.atomic_min(phi_local[i, j, k], d)

        # Update intersection counts
        j0 = ti.math.clamp(int(ti.ceil(ti.min(fjp, fjq, fjr))), 0, res_xyz[1] - 1)
        j1 = ti.math.clamp(int(ti.floor(ti.max(fjp, fjq, fjr))), 0, res_xyz[1] - 1)
        k0 = ti.math.clamp(int(ti.ceil(ti.min(fkp, fkq, fkr))), 0, res_xyz[2] - 1)
        k1 = ti.math.clamp(int(ti.floor(ti.max(fkp, fkq, fkr))), 0, res_xyz[2] - 1)

        for k in range(k0, k1 + 1):
            for j in range(j0, j1 + 1):
                a = ti.Vector([-1], ti.f32)
                b = ti.Vector([-1], ti.f32)
                c = ti.Vector([-1], ti.f32)

                if point_in_triangle_2d(j, k, fjp, fkp, fjq, fkq, fjr, fkr, a, b, c):
                    fi = a * fip + b * fiq + c * fir
                    i_interval = int(ti.ceil(fi))

                    if i_interval[0] < 0:
                        ti.atomic_add(intersection_count_local[0, j, k], 1)
                    elif i_interval[0] < res_xyz[0]:
                        ti.atomic_add(intersection_count_local[i_interval[0], j, k], 1)

@ti.kernel
def compute_sign(phi_local: ti.template(),
                 intersection_count_local: ti.template(),
                 res_xyz: vec3i):
    for k, j in ti.ndrange(res_xyz[2], res_xyz[1]):
        total_count = 0
        for i in range(res_xyz[0]):
            total_count += intersection_count_local[i, j, k]
            if total_count % 2 == 1:  # Odd parity means inside the mesh
                ti.atomic_add(phi_local[i, j, k], -2 * phi_local[i, j, k])  # Flip sign to indicate inside the mesh

@ti.kernel
def initialize_frontier(occupancy_local: ti.template(),
                        visited_local: ti.template(),
                        res_xyz: vec3i):
    """
    Initialize the frontier as the boundary voxels of the scene.
    """
    for i, j, k in ti.ndrange(res_xyz[0], res_xyz[1], res_xyz[2]):
        # Mark the outermost boundary voxels as "outside" (visited = 1)
        if (i == 0 or i == res_xyz[0] - 1 or j == 0 or j == res_xyz[1] - 1 or k == 0 or k == res_xyz[2] - 1) and occupancy_local[i, j, k] == 0:
            ti.atomic_max(visited_local[i, j, k], 1) # Mark as visited (outside)
            ti.atomic_min(occupancy_local[i, j, k], 0)  # Ensure boundary is set as unoccupied

@ti.kernel
def flood_fill_pass(occupancy_local: ti.template(),
                    visited_local: ti.template(),
                    res_xyz: vec3i):
    """
    Perform one iteration of the flood-fill algorithm.
    """
    for i, j, k in ti.ndrange(res_xyz[0], res_xyz[1], res_xyz[2]):
        if visited_local[i, j, k] == 1:
            # Expand to neighbors
            for di, dj, dk in ti.static(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                ni, nj, nk = i + di, j + dj, k + dk
                if 0 <= ni < res_xyz[0] and 0 <= nj < res_xyz[1] and 0 <= nk < res_xyz[2]:
                    if visited_local[ni, nj, nk] == 0 and occupancy_local[ni, nj, nk] == 0:
                        ti.atomic_max(visited_local[ni, nj, nk], 1)

@ti.kernel
def label_interior(occupancy_local: ti.template(),
                   visited_local: ti.template(),
                   res_xyz: vec3i):
    """
    Label all unvisited voxels as "inside" the object.
    """
    for i, j, k in ti.ndrange(res_xyz[0], res_xyz[1], res_xyz[2]):
        if visited_local[i, j, k] == 0:  # If the voxel was not visited by the flood fill, it's inside
            ti.atomic_max(occupancy_local[i, j, k], 1)  # Mark as "inside" the object
