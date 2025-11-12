#!/usr/bin/env python3
import numpy as np
import trimesh
from pathlib import Path
import shutil

# Paths
base = Path('outputs/fipt_real_classroom/export_optimized')
vt_path = base / 'vt.npy'
ft_path = base / 'ft.npy'
mesh_path = Path('data/fipt/real/classroom/scene.obj')
output_dir = Path('outputs/fipt_real_classroom/rhino_model_vflip')
output_dir.mkdir(parents=True, exist_ok=True)

# Load
mesh = trimesh.load(mesh_path)
vertices = mesh.vertices
faces = mesh.faces
vt = np.load(vt_path)
ft = np.load(ft_path)

# Flip V
vt_flipped = vt.copy()
vt_flipped[:,1] = 1.0 - vt_flipped[:,1]

# Write OBJ with flipped UVs
obj_path = output_dir / 'classroom_optimized_vflip.obj'
mtl_path = output_dir / 'classroom_optimized_vflip.mtl'

with open(obj_path, 'w') as f:
    f.write('# OBJ with V flipped\n')
    f.write(f'mtllib {mtl_path.name}\n\n')
    for v in vertices:
        f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
    f.write('\n')
    for uv in vt_flipped:
        f.write(f'vt {uv[0]:.6f} {uv[1]:.6f}\n')
    f.write('\nusemtl classroom_material_optimized\n')
    for i, face in enumerate(faces):
        uvf = ft[i]
        f.write(f'f {face[0]+1}/{uvf[0]+1} {face[1]+1}/{uvf[1]+1} {face[2]+1}/{uvf[2]+1}\n')

with open(mtl_path, 'w') as f:
    f.write('newmtl classroom_material_optimized\n')
    f.write('map_Kd albedo.png\n')

# copy textures
shutil.copy2('outputs/fipt_real_classroom/export_optimized/albedo.png', output_dir / 'albedo.png')
shutil.copy2('outputs/fipt_real_classroom/export_optimized/roughness.png', output_dir / 'roughness.png')
shutil.copy2('outputs/fipt_real_classroom/export_optimized/metallic.png', output_dir / 'metallic.png')

print('Written obj with V flipped to', obj_path)
print('Output dir:', output_dir)
