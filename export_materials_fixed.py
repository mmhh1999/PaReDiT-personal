#!/usr/bin/env python3
"""
Improved material export with better handling of coordinate alignment and invalid regions
"""

import mitsuba
mitsuba.set_variant('cuda_ad_rgb')

import os
import numpy as np
import torch
import trimesh
import xatlas
import nvdiffrast.torch as dr
from PIL import Image
from argparse import ArgumentParser
from model.brdf import NGPBRDF

def main():
    parser = ArgumentParser()
    parser.add_argument('--mesh', default='data/fipt/real/classroom/scene.obj')
    parser.add_argument('--ckpt', default='checkpoints/fipt_real_classroom/last_1.ckpt')
    parser.add_argument('--emitter_path', default='checkpoints/fipt_real_classroom/bake')
    parser.add_argument('--dir_save', default='outputs/fipt_real_classroom/export_fixed')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--tex_res', type=int, default=2048)
    parser.add_argument('--chunk_size', type=int, default=160000)
    parser.add_argument('--fix_invalid', action='store_true', default=True)
    args = parser.parse_args()

    dir_save = args.dir_save
    os.makedirs(dir_save, exist_ok=True)
    
    device = torch.device(args.device)
    
    # Load material network and get valid bounds
    mask = torch.load(os.path.join(args.emitter_path,'vslf.npz'), map_location='cpu', weights_only=False)
    voxel_min = mask['voxel_min']
    voxel_max = mask['voxel_max']
    
    material_net = NGPBRDF(voxel_min, voxel_max)
    state_dict = torch.load(args.ckpt, map_location='cpu', weights_only=False)['state_dict']
    weight = {}
    for k,v in state_dict.items():
        if 'material.' in k:
            weight[k.replace('material.','')]=v
    material_net.load_state_dict(weight)
    material_net.to(device)
    
    print(f'[INFO] loaded material network from {args.ckpt}')
    print(f'[INFO] valid bounds: min={voxel_min}, max={voxel_max}')

    # Load mesh
    mesh = trimesh.load(args.mesh)
    v_np = np.array(mesh.vertices).astype(np.float32)
    f_np = np.array(mesh.faces).astype(np.int32)
    
    print(f'[INFO] mesh bounds: min=[{v_np.min(0)}], max=[{v_np.max(0)}]')

    # Check for existing UVs or generate new ones
    path_ft_np = os.path.join(dir_save, 'ft.npy')
    path_vt_np = os.path.join(dir_save, 'vt.npy')
    
    if os.path.exists(path_ft_np) and os.path.exists(path_vt_np):
        print(f'[INFO] loading existing UVs from {path_ft_np} and {path_vt_np}')
        ft_np = np.load(path_ft_np)
        vt_np = np.load(path_vt_np)
    else:
        print(f'[INFO] generating UVs with xatlas for mesh: v={v_np.shape} f={f_np.shape}')
        atlas = xatlas.Atlas()
        atlas.add_mesh(v_np, f_np)
        
        chart_options = xatlas.ChartOptions()
        chart_options.max_iterations = 0
        pack_options = xatlas.PackOptions()
        pack_options.padding = 4  # Add padding to reduce seams
        atlas.generate(chart_options=chart_options, pack_options=pack_options)
        _, ft_np, vt_np = atlas[0]
        
        np.save(path_ft_np, ft_np)
        np.save(path_vt_np, vt_np)
    
    print(f'[INFO] UV mapping: v={v_np.shape} f={f_np.shape} vt={vt_np.shape} ft={ft_np.shape}')
    
    # Prepare for rasterization
    vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
    ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)
    
    # Convert UV to NDC coordinates [-1, 1]
    uv = vt * 2.0 - 1.0
    uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1)

    # Rasterization
    glctx = dr.RasterizeGLContext(output_db=False)
    tex_res = args.tex_res
    h, w = tex_res, tex_res
    
    rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), ft, (h, w))
    
    # Get 3D positions for each texel
    v = torch.from_numpy(v_np).to(device)
    f = torch.from_numpy(f_np).to(device)
    xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f)
    mask_render, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f)
    
    # Clean up GPU memory
    vt = vt.cpu()
    ft = ft.cpu()
    uv = uv.cpu()
    
    # Process positions
    xyzs = xyzs.view(-1, 3).cpu()
    mask_render = (mask_render > 0).view(-1).cpu()
    
    # Check which positions are within valid bounds
    voxel_min_np = voxel_min.cpu().numpy() if torch.is_tensor(voxel_min) else voxel_min
    voxel_max_np = voxel_max.cpu().numpy() if torch.is_tensor(voxel_max) else voxel_max
    
    mask_valid = mask_render.clone()
    if mask_render.any():
        pos_np = xyzs[mask_render].numpy()
        in_bounds = np.logical_and(
            np.all(pos_np >= voxel_min_np, axis=1),
            np.all(pos_np <= voxel_max_np, axis=1)
        )
        mask_valid[mask_render] = torch.from_numpy(in_bounds)
    
    invalid_ratio = 1.0 - mask_valid.float().mean().item()
    print(f'[INFO] invalid texel ratio: {invalid_ratio:.3f} ({invalid_ratio*100:.1f}%)')
    
    # Sample materials
    mat_dim = 5  # albedo(3) + roughness(1) + metallic(1)
    feats = torch.zeros(h * w, mat_dim, dtype=torch.float32).cpu()
    
    if mask_valid.any():
        with torch.no_grad():
            xyzs_valid = xyzs[mask_valid]
            
            # Set default values for invalid regions (light gray instead of black)
            default_albedo = torch.tensor([0.5, 0.5, 0.5])  # Gray instead of black
            default_roughness = torch.tensor([0.8])  # Slightly rough
            default_metallic = torch.tensor([0.0])   # Non-metallic
            
            chunk_size = args.chunk_size
            all_feats = torch.zeros((xyzs_valid.shape[0], mat_dim)).cpu()
            
            head = 0
            while head < xyzs_valid.shape[0]:
                tail = min(head + chunk_size, xyzs_valid.shape[0])
                
                slice_xyzs = xyzs_valid[head:tail].clone().detach().cuda()
                pred = material_net(slice_xyzs)
                
                # Extract material properties
                albedo = pred['albedo'].cpu()
                roughness = pred['roughness'].cpu()
                metallic = pred['metallic'].cpu()
                
                # Clamp values to reasonable ranges
                albedo = torch.clamp(albedo, 0.0, 1.0)
                roughness = torch.clamp(roughness, 0.0, 1.0)
                metallic = torch.clamp(metallic, 0.0, 1.0)
                
                slice_mats = torch.cat([albedo, roughness, metallic], dim=-1)
                all_feats[head:tail] = slice_mats.float()
                
                head += chunk_size
            
            feats[mask_valid] = all_feats
    
    # Set default values for invalid regions
    if not mask_valid.all():
        invalid_mask = ~mask_valid
        feats[invalid_mask, :3] = 0.5  # Gray albedo
        feats[invalid_mask, 3] = 0.8   # Roughness
        feats[invalid_mask, 4] = 0.0   # Metallic
    
    # Reshape and save
    feats = feats.view(h, w, -1)
    mask_render = mask_render.view(h, w)
    mask_valid = mask_valid.view(h, w)
    
    # Convert to 8-bit
    feats_np = feats.cpu().numpy()
    feats_np = np.clip(feats_np, 0.0, 1.0)
    feats_np = (feats_np * 255).astype(np.uint8)
    
    # Create separate texture maps
    albedo = feats_np[:, :, :3]
    roughness = feats_np[:, :, 3:4].repeat(3, axis=2)  # Make RGB for display
    metallic = feats_np[:, :, 4:5].repeat(3, axis=2)   # Make RGB for display
    
    # Save textures
    img_albedo = Image.fromarray(albedo)
    img_roughness = Image.fromarray(roughness)
    img_metallic = Image.fromarray(metallic)
    
    path_albedo = os.path.join(dir_save, 'albedo.png')
    path_roughness = os.path.join(dir_save, 'roughness.png')
    path_metallic = os.path.join(dir_save, 'metallic.png')
    
    img_albedo.save(path_albedo)
    img_roughness.save(path_roughness)
    img_metallic.save(path_metallic)
    
    # Save masks for debugging
    mask_render_img = Image.fromarray((mask_render.numpy() * 255).astype(np.uint8), 'L')
    mask_valid_img = Image.fromarray((mask_valid.numpy() * 255).astype(np.uint8), 'L')
    mask_render_img.save(os.path.join(dir_save, 'mask_render.png'))
    mask_valid_img.save(os.path.join(dir_save, 'mask_valid.png'))
    
    print(f'[INFO] saved textures:')
    print(f'  albedo: {path_albedo}')
    print(f'  roughness: {path_roughness}') 
    print(f'  metallic: {path_metallic}')
    print(f'[INFO] saved debug masks: mask_render.png, mask_valid.png')

if __name__ == '__main__':
    main()