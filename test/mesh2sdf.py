import os
import time
import trimesh
import argparse
import numpy as np
import ti_mesh2sdf
import taichi as ti
import skimage.measure

parser = argparse.ArgumentParser()
parser.add_argument("--obj", type=str, required=True)
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--exact_band", type=int, default=1)
parser.add_argument("--device", type=str, default='gpu')
args = parser.parse_args()

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

if args.device == 'gpu':
    device = ti.gpu
else:
    device = ti.cpu

ti.init(arch=device)

start = time.time()
print('Converting to watertight mesh...')
watertight_mesh = ti_mesh2sdf.compute_watertight_mesh(args.obj, args.resolution, args.exact_band)
print(f'Conversion took {time.time() - start:.2f}s')

ti.reset()
ti.init(arch=device)

start = time.time()
print('Converting watertight mesh to SDF...')
sdf = ti_mesh2sdf.compute_sdf_from_watertight(watertight_mesh, args.resolution, args.exact_band)
print(f'Conversion took {time.time() - start:.2f}s')

np.savez_compressed(os.path.join(output_dir, args.obj.split('/')[-1].split('.')[0] + '.sdf_res{}.npz'.format(args.resolution)), sdf=sdf)

vertices, faces, _, _ = skimage.measure.marching_cubes(sdf, 2 / args.resolution)
mesh = trimesh.Trimesh(vertices, faces)
mesh.export(os.path.join(output_dir, args.obj.split('/')[-1].split('.')[0] + '.sdf_res{}.obj'.format(args.resolution)))
