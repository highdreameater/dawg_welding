"""Blender (headless) step: decimate the two scanned rock patches and centre them on x/y, for tiling at reduced scale.

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python scripts/decimate_myscene_rocks.py

assets/myscene_export/ms_ground_rocks_18{,_001}.obj (349k tris each) -> ms_rocks_{a,b}_lod.obj (~15 %, UVs kept)
"""
import os

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "assets", "myscene_export")
RATIO = 0.15

for src, dst in (("ms_ground_rocks_18.obj", "ms_rocks_a_lod.obj"), ("ms_ground_rocks_18_001.obj", "ms_rocks_b_lod.obj")):
    bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete()
    bpy.ops.wm.obj_import(filepath=os.path.join(D, src), forward_axis="Y", up_axis="Z")
    ob = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = ob
    mod = ob.modifiers.new("dec", "DECIMATE"); mod.ratio = RATIO
    bpy.ops.object.modifier_apply(modifier=mod.name)
    xs = [v.co.x for v in ob.data.vertices]; ys = [v.co.y for v in ob.data.vertices]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    for v in ob.data.vertices:  # centre on x/y (keep z): scaling in MuJoCo then happens about the patch centre
        v.co.x -= cx; v.co.y -= cy
    bpy.ops.wm.obj_export(filepath=os.path.join(D, dst), export_selected_objects=True, export_materials=False,
                          export_normals=False, forward_axis="Y", up_axis="Z", export_triangulated_mesh=True)
    print(dst, "tris", len(ob.data.polygons), "centre was", round(cx, 3), round(cy, 3))
