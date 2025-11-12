#!/usr/bin/env python3
"""
Separate roughness and metallic channels for better Rhino3D compatibility
"""

from PIL import Image
import numpy as np

def separate_rm_channels(rm_path, output_dir):
    """
    Separate the rm.png file into individual roughness.png and metallic.png files
    """
    print(f"Loading combined texture: {rm_path}")
    rm_img = Image.open(rm_path)
    rm_array = np.array(rm_img)
    
    # Extract channels
    roughness = rm_array[:, :, 0]  # Red channel
    metallic = rm_array[:, :, 1]   # Green channel
    
    # Create grayscale images
    roughness_img = Image.fromarray(roughness, 'L')
    metallic_img = Image.fromarray(metallic, 'L')
    
    # Save separated textures
    roughness_path = f"{output_dir}/roughness.png"
    metallic_path = f"{output_dir}/metallic.png"
    
    roughness_img.save(roughness_path)
    metallic_img.save(metallic_path)
    
    print(f"✅ Saved roughness map: {roughness_path}")
    print(f"✅ Saved metallic map: {metallic_path}")
    
    return roughness_path, metallic_path

def create_improved_mtl(mtl_path, has_separate_channels=True):
    """
    Create an improved MTL file with separated texture channels
    """
    print(f"Creating improved MTL file: {mtl_path}")
    with open(mtl_path, 'w') as f:
        f.write("# IRIS Classroom Material - Enhanced for Rhino3D\n")
        f.write("# Generated with separated roughness and metallic maps\n\n")
        
        f.write("newmtl classroom_material\n")
        f.write("Ka 1.0 1.0 1.0\n")  # Ambient
        f.write("Kd 1.0 1.0 1.0\n")  # Diffuse  
        f.write("Ks 0.0 0.0 0.0\n")  # Specular
        f.write("Ns 1.0\n")          # Specular exponent
        f.write("d 1.0\n")           # Transparency (opaque)
        f.write("illum 2\n")         # Illumination model (color on, ambient on)
        
        # Texture maps
        f.write("map_Kd albedo.png\n")        # Diffuse/Base color
        
        if has_separate_channels:
            f.write("map_Pr roughness.png\n")     # Roughness (PBR)
            f.write("map_Pm metallic.png\n")      # Metallic (PBR)
        else:
            f.write("map_Pr rm.png\n")           # Combined roughness+metallic
            f.write("map_Pm rm.png\n")

if __name__ == "__main__":
    output_dir = "/home/ubuntu/PaReDiT-personal/outputs/fipt_real_classroom/rhino_model"
    rm_path = f"{output_dir}/rm.png"
    
    # Separate channels
    roughness_path, metallic_path = separate_rm_channels(rm_path, output_dir)
    
    # Create improved MTL file
    mtl_path = f"{output_dir}/classroom_with_materials.mtl"
    create_improved_mtl(mtl_path, has_separate_channels=True)
    
    print(f"\n🎉 Enhanced Rhino3D model ready!")
    print(f"📁 All files in: {output_dir}")
    print(f"📄 Texture files:")
    print(f"   - albedo.png (base color)")
    print(f"   - roughness.png (surface roughness)")
    print(f"   - metallic.png (metallic factor)")