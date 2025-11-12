#!/usr/bin/env python3
"""
Render quick camera previews of the textured mesh by sampling the baked atlas via rasterization.

Produces side-by-side comparisons: original camera image | textured render | absolute diff.
"""
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import nvdiffrast.torch as dr
import trimesh
import os

DATASET = Path('data/fipt/real/classroom')
EXPORT_DIR = Path('outputs/fipt_real_classroom/export_optimized')
PREVIEW_DIR = Path('outputs/fipt_real_classroom/preview')
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda'

def load_cameras():
    K_lines = open(DATASET / 'K_list.txt').read().strip().splitlines()
    Rt_lines = open(DATASET / 'Rt.txt').read().strip().splitlines()
    n = int(K_lines[0]); idx=1
    Ks=[]
    for i in range(n):
        K = np.array([list(map(float, K_lines[idx].split())), list(map(float, K_lines[idx+1].split())), list(map(float, K_lines[idx+2].split()))])
        Ks.append(K); idx+=3
    m = int(Rt_lines[0]); idx=1
    Rts=[]
    for i in range(m):
        mat = np.array([list(map(float, Rt_lines[idx].split())), list(map(float, Rt_lines[idx+1].split())), list(map(float, Rt_lines[idx+2].split())), list(map(float, Rt_lines[idx+3].split()))])
        Rts.append(mat); idx+=4
    return Ks, Rts

def build_expanded_face_buffers(verts, faces, vt, ft, K, Rt, H, W):
    # Build per-face-vertex buffers so UVs (which are per-corner) can be interpolated
    nfaces = faces.shape[0]
    # expanded vertices: 3 entries per face
    pos3 = np.zeros((nfaces*3, 3), dtype=np.float32)
    uv_attr = np.zeros((nfaces*3, 2), dtype=np.float32)
    face_idx = np.zeros((nfaces, 3), dtype=np.int32)

    for i in range(nfaces):
        for j in range(3):
            v_idx = faces[i, j]
            uv_idx = ft[i, j]
            pos3[i*3 + j] = verts[v_idx]
            uv_attr[i*3 + j] = vt[uv_idx]
            face_idx[i, j] = i*3 + j

    # world->cam
    world2cam = np.linalg.inv(Rt)
    R = world2cam[:3,:3]
    t = world2cam[:3,3]
    verts_cam = (R @ pos3.T).T + t[None,:]

    # project to pixel coords and create clip-like positions for rasterizer
    K = np.array(K)
    u_v = (K[0,0] * (verts_cam[:,0]/verts_cam[:,2]) + K[0,2])
    v_v = (K[1,1] * (verts_cam[:,1]/verts_cam[:,2]) + K[1,2])
    x_ndc = (u_v / (W-1)) * 2.0 - 1.0
    y_ndc = -((v_v / (H-1)) * 2.0 - 1.0)
    z = verts_cam[:,2]
    # pack to clip-like coordinates (x_ndc*z, y_ndc*z, z, z) so perspective divide recovers x_ndc,y_ndc
    pos_clip = np.stack([x_ndc * z, y_ndc * z, z, z], axis=1).astype(np.float32)
    return pos_clip, uv_attr.astype(np.float32), face_idx

def render_preview(vt_path, ft_path, atlas_path, name_tag='default'):
    device = torch.device(DEVICE)
    mesh = trimesh.load(str(DATASET / 'scene.obj'))
    verts = np.array(mesh.vertices).astype(np.float32)
    faces = np.array(mesh.faces).astype(np.int32)

    vt = np.load(vt_path).astype(np.float32)
    ft = np.load(ft_path).astype(np.int32)
    atlas = np.array(Image.open(atlas_path).convert('RGB'))
    H_t, W_t = atlas.shape[:2]

    Ks, Rts = load_cameras()
    imgs = sorted([p for p in (DATASET/'Image').iterdir() if p.suffix.lower()=='.png'])

    glctx = dr.RasterizeGLContext()

    # choose a small set of cameras to preview (evenly spaced)
    n = len(Ks)
    picks = [0, n//4, n//2, 3*n//4] if n>=4 else list(range(n))

    for idx in picks:
        K = Ks[idx]; Rt = Rts[idx]
        imgp = imgs[idx]
        img = np.array(Image.open(imgp).convert('RGB'))
        H, W = img.shape[:2]

        pos_clip, uv_attr, face_idx = build_expanded_face_buffers(verts, faces, vt, ft, K, Rt, H, W)
        pos_t = torch.from_numpy(pos_clip).to(device).contiguous()
        faces_t = torch.from_numpy(face_idx.astype(np.int32)).to(device).contiguous()

        rast, _ = dr.rasterize(glctx, pos_t.unsqueeze(0), faces_t, (H, W))

        # interpolate uv attribute from per-face-vertex uv_attr
        uv_attr_t = torch.from_numpy(uv_attr).to(device).unsqueeze(0).contiguous()
        uv_pix, _ = dr.interpolate(uv_attr_t, rast, faces_t)
        uv_map = uv_pix[0].cpu().numpy()
        # uv_map shape HxWx2 (might be padded channels)
        # Some pixels may have invalid uv (rast channels indicate coverage). Use prim_id channel
        prim_id = rast[0,...,3].cpu().numpy().astype(np.int32)
        covered = prim_id != 0

        # sample atlas at uv_map
        u_pix = np.clip((uv_map[...,0] * (W_t-1)).astype(np.int32), 0, W_t-1)
        v_pix = np.clip((uv_map[...,1] * (H_t-1)).astype(np.int32), 0, H_t-1)
        sampled = np.zeros_like(img)
        sampled[covered] = atlas[v_pix[covered], u_pix[covered]]

        # compose side-by-side
        diff = np.abs(img.astype(np.int32) - sampled.astype(np.int32)).astype(np.uint8)
        side = np.concatenate([img, sampled, diff], axis=1)
        outp = PREVIEW_DIR / f'preview_{name_tag}_cam{idx:03d}.png'
        Image.fromarray(side).save(outp)
        print('Saved preview', outp)

def main():
    ft_path = EXPORT_DIR / 'ft.npy'
    vt_orig = EXPORT_DIR / 'vt.npy'
    vt_flipped = EXPORT_DIR / 'vt_flipped.npy'
    atlas = EXPORT_DIR / 'albedo.png'

    # Render both original and flipped if available
    if vt_orig.exists():
        render_preview(vt_orig, ft_path, atlas, name_tag='orig')
    if vt_flipped.exists():
        render_preview(vt_flipped, ft_path, atlas, name_tag='flipped')

if __name__ == '__main__':
    main()
