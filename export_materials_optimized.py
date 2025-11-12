#!/usr/bin/env python3
"""
Optimized UV mapping with better space utilization
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
    parser.add_argument('--dir_save', default='outputs/fipt_real_classroom/export_optimized')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--tex_res', type=int, default=2048)
    parser.add_argument('--chunk_size', type=int, default=160000)
    parser.add_argument('--flip_v', action='store_true', help='Flip V coordinate of UVs before baking (writes vt_flipped.npy)')
    parser.add_argument('--write_obj', action='store_true', help='Write a Rhino-compatible OBJ/MTL into dir_save using the baked textures')
    args = parser.parse_args()

    dir_save = args.dir_save
    os.makedirs(dir_save, exist_ok=True)
    
    device = torch.device(args.device)
    
    # Load material network
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

    # Load mesh
    mesh = trimesh.load(args.mesh)
    v_np = np.array(mesh.vertices).astype(np.float32)
    f_np = np.array(mesh.faces).astype(np.int32)
    
    print(f'[INFO] mesh: {v_np.shape[0]:,} vertices, {f_np.shape[0]:,} faces')

    # Generate optimized UV mapping with xatlas
    print(f'[INFO] generating optimized UV mapping...')
    atlas = xatlas.Atlas()
    atlas.add_mesh(v_np, f_np)
    
    # Optimize chart generation for better packing
    chart_options = xatlas.ChartOptions()
    # Adjust chart sizing to avoid excessive fragmentation. Larger chart areas
    # produce fewer, bigger UV islands which reduces the "fragmented" look
    # in the baked textures. The previous setting (1/64) created many tiny
    # charts that packed into a small corner of the atlas.
    chart_options.max_iterations = 4  # Allow merge iterations
    chart_options.max_chart_area = 1.0/8.0  # Larger charts for less fragmentation
    chart_options.max_boundary_length = 0.0  # No boundary length limit
    chart_options.normal_deviation_weight = 2.0  # Prefer normal alignment
    chart_options.roundness_weight = 0.01  # Less emphasis on roundness
    chart_options.straightness_weight = 6.0  # Prefer straight boundaries
    chart_options.normal_seam_weight = 4.0  # Avoid seams across normal boundaries
    chart_options.texture_seam_weight = 0.5  # Lower texture seam weight
    
    # Optimize packing
    pack_options = xatlas.PackOptions()
    pack_options.padding = 2  # Smaller padding for tighter packing
    pack_options.texels_per_unit = 0.0  # Auto texel density
    pack_options.resolution = args.tex_res
    pack_options.bilinear = True  # Enable bilinear filtering
    pack_options.blockAlign = True  # Align to block boundaries
    pack_options.bruteForce = False  # Don't use brute force (faster)
    
    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    _, ft_np, vt_np = atlas[0]
    
    print(f'[INFO] UV generation complete: {vt_np.shape[0]:,} UV vertices')
    
    # Save UV data
    np.save(os.path.join(dir_save, 'ft.npy'), ft_np)
    np.save(os.path.join(dir_save, 'vt.npy'), vt_np)
    if args.flip_v:
        vt_flipped = vt_np.copy()
        vt_flipped[:,1] = 1.0 - vt_flipped[:,1]
        np.save(os.path.join(dir_save, 'vt_flipped.npy'), vt_flipped)
        print(f'[INFO] saved flipped UVs to {os.path.join(dir_save, "vt_flipped.npy")}')
    
    # Calculate UV utilization before rasterization
    vt_analysis = vt_np.copy()
    print(f'[INFO] UV coordinate analysis:')
    print(f'  U range: [{vt_analysis[:, 0].min():.3f}, {vt_analysis[:, 0].max():.3f}]')
    print(f'  V range: [{vt_analysis[:, 1].min():.3f}, {vt_analysis[:, 1].max():.3f}]')
    
    # Setup for rasterization
    vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
    ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)
    
    # Optionally flip V to match different OBJ/renderer conventions
    if args.flip_v:
        print('[INFO] flipping V coordinate for UVs before rasterization')
        vt = vt.clone()
        vt[:,1] = 1.0 - vt[:,1]

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
    
    # Calculate UV efficiency
    uv_efficiency = mask_render.float().mean().item()
    print(f'[INFO] UV mapping efficiency: {uv_efficiency:.3f} ({uv_efficiency*100:.1f}%)')
    
    # Sample materials only for valid regions
    mat_dim = 5
    feats = torch.zeros(h * w, mat_dim, dtype=torch.float32).cpu()
    
    if mask_render.any():
        with torch.no_grad():
            xyzs_valid = xyzs[mask_render]
            
            chunk_size = args.chunk_size
            all_feats = torch.zeros((xyzs_valid.shape[0], mat_dim)).cpu()
            
            head = 0
            while head < xyzs_valid.shape[0]:
                tail = min(head + chunk_size, xyzs_valid.shape[0])
                
                slice_xyzs = xyzs_valid[head:tail].clone().detach().cuda()
                
                try:
                    pred = material_net(slice_xyzs)
                    albedo = torch.clamp(pred['albedo'].cpu(), 0.01, 0.99)  # Avoid pure black/white
                    roughness = torch.clamp(pred['roughness'].cpu(), 0.01, 0.99)
                    metallic = torch.clamp(pred['metallic'].cpu(), 0.0, 1.0)
                except Exception as e:
                    print(f'[WARN] Material sampling failed for chunk {head}-{tail}: {e}')
                    # Use default materials for failed chunks
                    albedo = torch.full((tail-head, 3), 0.5)
                    roughness = torch.full((tail-head, 1), 0.7)
                    metallic = torch.full((tail-head, 1), 0.1)
                
                slice_mats = torch.cat([albedo, roughness, metallic], dim=-1)
                all_feats[head:tail] = slice_mats.float()
                
                head += chunk_size
            
            feats[mask_render] = all_feats
    
    # Set reasonable defaults for empty regions (will be mostly invisible anyway)
    empty_mask = ~mask_render
    feats[empty_mask, :3] = 0.2  # Dark gray for unused areas
    feats[empty_mask, 3] = 0.9   # High roughness
    feats[empty_mask, 4] = 0.0   # Non-metallic
    
    # Reshape and convert
    feats = feats.view(h, w, -1)
    mask_render = mask_render.view(h, w)
    
    feats_np = feats.cpu().numpy()
    feats_np = np.clip(feats_np, 0.0, 1.0)
    feats_np = (feats_np * 255).astype(np.uint8)
    
    # Create texture maps
    albedo = feats_np[:, :, :3]
    roughness_single = feats_np[:, :, 3]
    metallic_single = feats_np[:, :, 4]
    
    # Save main textures
    img_albedo = Image.fromarray(albedo)
    img_roughness = Image.fromarray(roughness_single, 'L')
    img_metallic = Image.fromarray(metallic_single, 'L')
    
    path_albedo = os.path.join(dir_save, 'albedo.png')
    path_roughness = os.path.join(dir_save, 'roughness.png')
    path_metallic = os.path.join(dir_save, 'metallic.png')
    
    img_albedo.save(path_albedo)
    img_roughness.save(path_roughness)
    img_metallic.save(path_metallic)
    
    # Save usage mask
    usage_mask = Image.fromarray((mask_render.numpy() * 255).astype(np.uint8), 'L')
    usage_mask.save(os.path.join(dir_save, 'uv_usage_mask.png'))
    
    print(f'[INFO] optimized textures saved:')
    print(f'  albedo: {path_albedo}')
    print(f'  roughness: {path_roughness}')
    print(f'  metallic: {path_metallic}')
    print(f'  UV efficiency: {uv_efficiency*100:.1f}%')

if __name__ == '__main__':
    main()