#!/usr/bin/env python3
"""
Create a Rhino-compatible 3D model with materials from IRIS output
"""

import numpy as np
import trimesh
import os
from pathlib import Path

def create_obj_with_materials(mesh_path, vt_path, ft_path, albedo_path, rm_path, output_dir):
    """
    Create OBJ + MTL files with UV coordinates and material textures for Rhino3D
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load original mesh
    print(f"Loading mesh from {mesh_path}")
    mesh = trimesh.load(mesh_path)
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Load UV coordinates
    print(f"Loading UV coordinates...")
    vt = np.load(vt_path)  # UV coordinates
    ft = np.load(ft_path)  # Face UV indices
    
    print(f"Mesh info:")
    print(f"  Vertices: {vertices.shape}")
    print(f"  Faces: {faces.shape}")
    print(f"  UV vertices: {vt.shape}")
    print(f"  UV faces: {ft.shape}")
    
    # Create OBJ file
    obj_path = output_dir / "classroom_with_materials.obj"
    mtl_path = output_dir / "classroom_with_materials.mtl"
    
    print(f"Creating OBJ file: {obj_path}")
    with open(obj_path, 'w') as f:
        # Header
        f.write("# IRIS Classroom Model with Materials\n")
        f.write("# Generated for Rhino3D\n")
        f.write(f"mtllib {mtl_path.name}\n\n")
        
        # Write vertices
        f.write("# Vertices\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write UV coordinates
        f.write("\n# UV Coordinates\n")
        for uv in vt:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        
        # Write material group and faces
        f.write(f"\n# Material\n")
        f.write("usemtl classroom_material\n")
        f.write("# Faces\n")
        for i, face in enumerate(faces):
            uv_face = ft[i]
            # OBJ uses 1-based indexing
            f.write(f"f {face[0]+1}/{uv_face[0]+1} {face[1]+1}/{uv_face[1]+1} {face[2]+1}/{uv_face[2]+1}\n")
    
    # Create MTL file
    print(f"Creating MTL file: {mtl_path}")
    with open(mtl_path, 'w') as f:
        f.write("# IRIS Material for Classroom\n")
        f.write("newmtl classroom_material\n")
        f.write("Ka 1.0 1.0 1.0\n")  # Ambient
        f.write("Kd 1.0 1.0 1.0\n")  # Diffuse
        f.write("Ks 0.0 0.0 0.0\n")  # Specular
        f.write("Ns 1.0\n")          # Specular exponent
        f.write("d 1.0\n")           # Transparency
        f.write("illum 2\n")         # Illumination model
        
        # Texture maps
        f.write(f"map_Kd albedo.png\n")        # Diffuse texture (albedo)
        f.write(f"map_Pr rm.png\n")           # Roughness map (red channel)
        f.write(f"map_Pm rm.png\n")           # Metallic map (green channel)
    
    # Copy texture files
    import shutil
    albedo_dest = output_dir / "albedo.png"
    rm_dest = output_dir / "rm.png"
    
    print(f"Copying textures...")
    shutil.copy2(albedo_path, albedo_dest)
    shutil.copy2(rm_path, rm_dest)
    
    # Create Rhino import instructions
    instructions_path = output_dir / "RHINO_IMPORT_INSTRUCTIONS.txt"
    with open(instructions_path, 'w') as f:
        f.write("IRIS Classroom Model - Rhino3D Import Instructions\n")
        f.write("=" * 50 + "\n\n")
        f.write("Files included:\n")
        f.write("- classroom_with_materials.obj (3D geometry with UV mapping)\n")
        f.write("- classroom_with_materials.mtl (material definitions)\n")
        f.write("- albedo.png (diffuse color texture, 2048x2048)\n")
        f.write("- rm.png (roughness=red, metallic=green channels, 2048x2048)\n\n")
        
        f.write("Import Steps:\n")
        f.write("1. Open Rhino3D\n")
        f.write("2. Use 'Import' command and select 'classroom_with_materials.obj'\n")
        f.write("3. Make sure 'Import materials' is checked\n")
        f.write("4. The model will be imported with UV coordinates and materials\n\n")
        
        f.write("Material Setup:\n")
        f.write("1. Open Material Editor (Properties panel > Materials)\n")
        f.write("2. Find 'classroom_material'\n")
        f.write("3. Verify texture paths are correct\n")
        f.write("4. For PBR rendering:\n")
        f.write("   - Base Color: albedo.png\n")
        f.write("   - Roughness: rm.png (red channel)\n")
        f.write("   - Metallic: rm.png (green channel)\n\n")
        
        f.write("Rendering:\n")
        f.write("- Use Rhino Render or Cycles for best results\n")
        f.write("- Enable 'Use material assignments' in render settings\n")
        f.write("- For Cycles: Set material to 'Principled BSDF'\n\n")
        
        f.write("Note: The rm.png file contains both roughness (red) and metallic (green)\n")
        f.write("values. You may need to separate these into individual texture files\n")
        f.write("depending on your Rhino version and renderer.\n")
    
    print(f"\n✅ Rhino3D model created successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Files created:")
    print(f"   - {obj_path.name} (3D model)")
    print(f"   - {mtl_path.name} (materials)")
    print(f"   - albedo.png (diffuse texture)")
    print(f"   - rm.png (roughness + metallic)")
    print(f"   - {instructions_path.name} (import guide)")

if __name__ == "__main__":
    # Paths for classroom scene
    mesh_path = "/home/ubuntu/PaReDiT-personal/data/fipt/real/classroom/scene.obj"
    vt_path = "/home/ubuntu/PaReDiT-personal/outputs/fipt_real_classroom/export/vt.npy"
    ft_path = "/home/ubuntu/PaReDiT-personal/outputs/fipt_real_classroom/export/ft.npy"
    albedo_path = "/home/ubuntu/PaReDiT-personal/outputs/fipt_real_classroom/export/albedo.png"
    rm_path = "/home/ubuntu/PaReDiT-personal/outputs/fipt_real_classroom/export/rm.png"
    output_dir = "/home/ubuntu/PaReDiT-personal/outputs/fipt_real_classroom/rhino_model"
    
    create_obj_with_materials(mesh_path, vt_path, ft_path, albedo_path, rm_path, output_dir)