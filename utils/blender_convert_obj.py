# From: https://github.com/One-2-3-45/One-2-3-45/blob/master/render/single_render_eval.py

import blenderproc as bproc

import os
import time
import math
import argparse

"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.
"""

import bpy
from mathutils import Vector

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="example")
parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
parser.add_argument("--object_path", type=str, required=True, help="Path to the object file")
args = parser.parse_args()

bproc.init()

context = bpy.context
scene = context.scene
render = scene.render
render.engine = args.engine


# load the glb model
def load_object(object_path: str, output_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
        normalize_scene()
        bpy.ops.wm.obj_export(filepath=output_path)

        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()


        bpy.ops.wm.obj_import(filepath=output_path)
        mesh = bpy.context.active_object
        bpy.ops.object.join()
        bpy.ops.wm.obj_export(filepath=output_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene(format="glb"):
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


if __name__ == "__main__":
    try:
        start_i = time.time()
        output_dir = args.output_dir
        model_id = args.object_path.split('/')[-1].split('.')[0]
        output_path = os.path.join(output_dir, '{}.obj'.format(model_id))
        load_object(args.object_path, output_path=output_path)
        end_i = time.time()
        print("Finished", args.object_path, "in", end_i - start_i, "seconds")
    except Exception as e:
        print("Failed to convert", args.object_path)
        print(e)
