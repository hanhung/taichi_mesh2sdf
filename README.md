# Taichi Mesh2SDF

A Taichi-based reimplementation of [mesh2sdf](https://github.com/wang-ps/mesh2sdf), with occupancy conversion and flood-filling. Note that the original sweeping algorithm is not included, as it is difficult to parallelize. Instead, flood filling is used to for the occupancy grid. Unlike the original mesh2sdf, which assumes a cubic grid, this version allows for non-uniform grid lengths along the x, y, and z axes to reduce unnecessary computation.

## Installation

1. Install Package
```
pip install .
```

## Usage

**[Notes]:** Curently only works on CPU and metal (cuda gives wrong results for mesh2sdf). There is also a limitation on the maximum resolution. As taichi only allows for 32 bit indexing.

1. Download and normalize object
```
python utils/download_example.py
blenderproc run utils/blender_convert_obj.py --object_path=example/glbs/000-138/3b61335c2a004a9ea31c8dab59471222.glb
```
2. Convert mesh to sdf
```
python test/mesh2sdf.py --obj=example/3b61335c2a004a9ea31c8dab59471222.obj --resolution=1162 --device=cpu
```
3. Convert sdf to occ and flood fill
```
python test/sdf2occ.py --obj=output/3b61335c2a004a9ea31c8dab59471222.sdf_res1162.npz --device=gpu
```
4. You can find the results in ***./output***.

## Notes

For each scene in [nuiscene43](https://huggingface.co/datasets/3dlg-hcvc/NuiScene43/tree/main/nuiscene43) you will find file names such as ***3b61335c2a004a9ea31c8dab59471222.occ_res1162.npz*** or ***5fc65fd24ca647388d055dbc122b2c53.occ_res1393.npz***. This number indicates the resolution of the occupancy grid used to process the scene. We used the labeled scale for each scene to scale them accordingly during the occupancy conversion process. This results in scenes that are naturally unified in scales after the occupancy conversion.

## Acknowledgement

Thanks to the authors of [mesh2sdf](https://github.com/wang-ps/mesh2sdf) for opensourcing their code. If you use this tool please cite their awesome work as well as ours.

```
@article {Wang-SIG2022,
  title      = {Dual Octree Graph Networks
                for Learning Adaptive Volumetric Shape Representations},
  author     = {Wang, Peng-Shuai and Liu, Yang and Tong, Xin},
  journal    = {ACM Transactions on Graphics (SIGGRAPH)},
  volume     = {41},
  number     = {4},
  year       = {2022},
}
```

```
@article{lee2025nuiscene,
  title={NuiScene: Exploring efficient generation of unbounded outdoor scenes},
  author={Lee, Han-Hung and Han, Qinghong and Chang, Angel X},
  journal={arXiv preprint arXiv:2503.16375},
  year={2025}
}
```
