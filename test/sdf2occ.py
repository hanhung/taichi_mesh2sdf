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
parser.add_argument("--device", type=str, default='gpu')
args = parser.parse_args()

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

if args.device == 'gpu':
    device = ti.gpu
else:
    device = ti.cpu

ti.init(arch=device)

resolution = int(args.obj.split('/')[-1].split('.')[1].split('_res')[-1])
sdf = np.load(args.obj)['sdf']
occ = (sdf <= 2 / resolution).astype(np.int32)

start = time.time()
print('Flood filling occupancy...')
occ = ti_mesh2sdf.compute_occ_flood_fill(occ)
print(f'Flood filling took took {time.time() - start:.2f}s')

np.savez_compressed(os.path.join(output_dir, args.obj.split('/')[-1].split('.')[0] + '.occ_res{}.npz'.format(resolution)), occ=occ)

vertices, faces, _, _ = skimage.measure.marching_cubes(occ, 0)
mesh = trimesh.Trimesh(vertices, faces)
mesh.export(os.path.join(output_dir, args.obj.split('/')[-1].split('.')[0] + '.occ_res{}.obj'.format(resolution)))
