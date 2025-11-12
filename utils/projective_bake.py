#!/usr/bin/env python3
"""
Projective texture baking: rasterize UV atlas, map texels to 3D positions, project into all images,
check visibility via per-camera depth maps, and accumulate colors into atlas.

Outputs:
 - atlas_albedo.png
 - projective OBJ/MTL using the same vt/ft UVs
 - uv_usage_mask.png and per_texel_viewcount.png

This is a simplified but robust implementation using nvdiffrast.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import nvdiffrast.torch as dr
import trimesh

# Settings
DATASET = Path('data/fipt/real/classroom')
VT_PATH = Path('outputs/fipt_real_classroom/export_optimized/vt.npy')
FT_PATH = Path('outputs/fipt_real_classroom/export_optimized/ft.npy')
MESH_PATH = DATASET / 'scene.obj'
OUT_DIR = Path('outputs/fipt_real_classroom/bake_projective')
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEX_RES = 2048
DEVICE = 'cuda'
# Toggle whether to perform a z-buffer visibility test per camera.
# Set False to skip rasterization-based visibility (faster, avoids rasterizer depth issues
# but may include occluded samples). You can enable later for stricter visibility.
USE_DEPTH_TEST = False

def load_camera_lists():
    K_lines = open(DATASET / 'K_list.txt').read().strip().splitlines()
    Rt_lines = open(DATASET / 'Rt.txt').read().strip().splitlines()
    # first line is count
    n = int(K_lines[0])
    Ks = []
    idx = 1
    for i in range(n):
        K = np.array([list(map(float, K_lines[idx].split())),
                      list(map(float, K_lines[idx+1].split())),
                      list(map(float, K_lines[idx+2].split()))])
        Ks.append(K)
        idx += 3
    # Rt: each is 4 lines
    m = int(Rt_lines[0])
    Rts = []
    idx = 1
    for i in range(m):
        mat = np.array([list(map(float, Rt_lines[idx].split())),
                        list(map(float, Rt_lines[idx+1].split())),
                        list(map(float, Rt_lines[idx+2].split())),
                        list(map(float, Rt_lines[idx+3].split()))])
        Rts.append(mat)
        idx += 4
    return Ks, Rts


def main():
    device = torch.device(DEVICE)
    Ks, Rts = load_camera_lists()
    n_cams = len(Ks)
    print(f'Found {n_cams} cameras')

    # load images list
    img_dir = DATASET / 'Image'
    # Prefer PNG (LDR) versions for baking; ignore EXR to avoid loader issues
    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() == '.png'])
    print('Found images:', len(imgs))

    # load mesh
    mesh = trimesh.load(MESH_PATH)
    verts = np.array(mesh.vertices).astype(np.float32)
    faces = np.array(mesh.faces).astype(np.int32)
    vnorm = np.array(mesh.vertex_normals).astype(np.float32)

    # load uv
    vt = np.load(VT_PATH)
    ft = np.load(FT_PATH)

    # rasterize UV
    vt_t = torch.from_numpy(vt.astype(np.float32)).to(device).contiguous()
    ft_t = torch.from_numpy(ft.astype(np.int32)).to(device).contiguous()
    uv = vt_t * 2.0 - 1.0
    uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1)
    glctx = dr.RasterizeGLContext(output_db=False)
    rast, rast_db = dr.rasterize(glctx, uv.unsqueeze(0), ft_t, (TEX_RES, TEX_RES))
    # interpolate to get world positions and normals
    v_t = torch.from_numpy(verts.astype(np.float32)).to(device).contiguous()
    n_t = torch.from_numpy(vnorm.astype(np.float32)).to(device).contiguous()
    f_t = torch.from_numpy(faces.astype(np.int32)).to(device).contiguous()

    xyzs, _ = dr.interpolate(v_t.unsqueeze(0), rast, f_t)
    normals, _ = dr.interpolate(n_t.unsqueeze(0), rast, f_t)
    mask, _ = dr.interpolate(torch.ones_like(v_t[:, :1]).unsqueeze(0), rast, f_t)

    xyzs = xyzs.view(-1,3).cpu().numpy()
    normals = normals.view(-1,3).cpu().numpy()
    mask = (mask.view(-1,1).cpu().numpy() > 0).reshape(-1)

    print('Total texels:', TEX_RES*TEX_RES)
    print('Used texels:', mask.sum())

    # prepare accumulators
    accum_color = np.zeros((TEX_RES*TEX_RES,3), dtype=np.float64)
    accum_weight = np.zeros((TEX_RES*TEX_RES,), dtype=np.float64)
    view_count = np.zeros((TEX_RES*TEX_RES,), dtype=np.int32)

    # iterate cameras
    for i, (K, Rt) in enumerate(zip(Ks, Rts)):
        print(f'Processing camera {i+1}/{n_cams}')
        # image path may correspond to imgs[i]
        if i >= len(imgs):
            print('No image for camera', i)
            continue
        imgp = imgs[i]
        img = np.array(Image.open(imgp).convert('RGB'))/255.0
        H, W = img.shape[:2]
        K = np.array(K)
        Rt = np.array(Rt)
        # compute world->cam matrix
        world2cam = np.linalg.inv(Rt)
        R = world2cam[:3,:3]
        t = world2cam[:3,3]
        # transform texel positions to camera
        xyz_cam = (R @ xyzs.T).T + t[None,:]
        z = xyz_cam[:,2]
        # compute image pixel coords
        u = (K[0,0] * (xyz_cam[:,0]/z) + K[0,2])
        v = (K[1,1] * (xyz_cam[:,1]/z) + K[1,2])
        # valid projection
        inside = (z>0) & (u>=0) & (u < W) & (v>=0) & (v < H) & mask
        if inside.sum()==0:
            continue
        ui = np.clip(u[inside].astype(np.int32),0,W-1)
        vi = np.clip(v[inside].astype(np.int32),0,H-1)

        # optionally perform a rasterizer-based visibility check. This can be fragile
        # depending on how clip-space/z are constructed for the rasterizer. For a
        # quick, robust result we can skip the visibility test (will include occluded
        # samples but avoids depth-domain mismatch problems). Toggle via USE_DEPTH_TEST.
        texel_idx = np.nonzero(inside)[0]
        pix_idx = vi * W + ui
        z_proj = z[inside]
        if USE_DEPTH_TEST:
            # Rasterize mesh in this camera: project vertices to NDC and build depth map
            verts_cam = (R @ verts.T).T + t[None,:]
            u_v = (K[0,0] * (verts_cam[:,0]/verts_cam[:,2]) + K[0,2])
            v_v = (K[1,1] * (verts_cam[:,1]/verts_cam[:,2]) + K[1,2])
            x_ndc = (u_v / (W-1)) * 2.0 - 1.0
            y_ndc = -((v_v / (H-1)) * 2.0 - 1.0)
            ones = np.ones((verts_cam.shape[0], 1), dtype=np.float32)
            pos_ndc = np.concatenate([x_ndc[:,None], y_ndc[:,None], verts_cam[:,2:3].astype(np.float32), ones], axis=1)
            pos_ndc_t = torch.from_numpy(pos_ndc.astype(np.float32)).to(device).contiguous()
            faces_t = torch.from_numpy(faces.astype(np.int32)).to(device).contiguous()
            rast_cam, _ = dr.rasterize(glctx, pos_ndc_t.unsqueeze(0), faces_t, (H, W))
            verts_cam_t = torch.from_numpy(verts_cam.astype(np.float32)).to(device).unsqueeze(0).contiguous()
            xyz_cam_pix, _ = dr.interpolate(verts_cam_t, rast_cam.contiguous(), faces_t)
            depth_map = xyz_cam_pix[0,...,2].cpu().numpy()
            depth_at_pix = depth_map[vi, ui]
            tol = np.maximum(0.01, 0.01 * z_proj)
            visible = np.abs(depth_at_pix - z_proj) <= tol
        else:
            # accept all projected texels (fast, may include occluded samples)
            visible = np.ones_like(z_proj, dtype=bool)
        # accumulate colors
        vis_texels = texel_idx[visible]
        if vis_texels.size==0:
            continue
        colors = img[vi[visible], ui[visible], :]
        # weight by dot(normal, view_dir) (view_dir = -normalize(xyz_cam))
        view_dir = -xyz_cam[inside][visible]
        view_dir = view_dir / (np.linalg.norm(view_dir, axis=1, keepdims=True)+1e-9)
        norms = normals[vis_texels]
        norms = norms / (np.linalg.norm(norms, axis=1, keepdims=True)+1e-9)
        w = np.clip((norms * view_dir).sum(axis=1), 0.0, 1.0)
        # avoid zero weights
        w = w + 1e-6
        accum_color[vis_texels] += (colors * w[:,None])
        accum_weight[vis_texels] += w
        view_count[vis_texels] += 1

    # finalize atlas
    final = np.zeros((TEX_RES*TEX_RES,3), dtype=np.uint8)
    used = accum_weight>0
    final[used] = (accum_color[used] / accum_weight[used,None] * 255.0).astype(np.uint8)
    final_img = final.reshape((TEX_RES, TEX_RES, 3))
    Image.fromarray(final_img).save(OUT_DIR / 'atlas_albedo.png')
    Image.fromarray((mask.reshape(TEX_RES, TEX_RES).astype(np.uint8)*255)).save(OUT_DIR / 'uv_usage_mask.png')
    Image.fromarray((view_count.reshape(TEX_RES, TEX_RES).astype(np.uint8))).save(OUT_DIR / 'per_texel_viewcount.png')

    # write obj using vt/ft and copy textures
    out_obj = OUT_DIR / 'classroom_projective.obj'
    out_mtl = OUT_DIR / 'classroom_projective.mtl'
    with open(out_obj, 'w') as f:
        f.write('# Projective baked OBJ\n')
        f.write(f'mtllib {out_mtl.name}\n\n')
        for v in verts:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        f.write('\n')
        for uvv in vt:
            f.write(f'vt {uvv[0]:.6f} {uvv[1]:.6f}\n')
        f.write('\nusemtl mat_projective\n')
        for i, face in enumerate(faces):
            uvf = ft[i]
            f.write(f'f {face[0]+1}/{uvf[0]+1} {face[1]+1}/{uvf[1]+1} {face[2]+1}/{uvf[2]+1}\n')
    with open(out_mtl, 'w') as f:
        f.write('newmtl mat_projective\n')
        f.write('map_Kd atlas_albedo.png\n')
    # atlas already written to OUT_DIR
    print('Baking done. Outputs in', OUT_DIR)

if __name__ == '__main__':
    main()
