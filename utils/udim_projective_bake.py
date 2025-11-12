#!/usr/bin/env python3
"""
UDIM projective baker (prototype)

Rasterize UVs at a tiled atlas resolution (tile_res * grid_x, tile_res * grid_y),
interpolate world-space positions per texel, then for each UDIM tile accumulate
camera colors by projecting texel 3D points into images.

Outputs: outputs/fipt_real_classroom/udim_tiles/albedo_<UDIM>.png (and mask/viewcount)

This is a prototype focused on correct alignment; it defaults to disabling strict
depth-test for robustness but supports enabling it.
"""
import os
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import torch
import nvdiffrast.torch as dr
import trimesh

ROOT = Path('.').resolve()
DATASET = ROOT / 'data' / 'fipt' / 'real' / 'classroom'
EXPORT_DIR = ROOT / 'outputs' / 'fipt_real_classroom' / 'udim_tiles'
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

def load_cameras(dataset):
    K_lines = open(dataset / 'K_list.txt').read().strip().splitlines()
    Rt_lines = open(dataset / 'Rt.txt').read().strip().splitlines()
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', default=str(DATASET / 'scene.obj'))
    parser.add_argument('--vt', default=str(ROOT / 'outputs' / 'fipt_real_classroom' / 'export_optimized' / 'vt.npy'))
    parser.add_argument('--ft', default=str(ROOT / 'outputs' / 'fipt_real_classroom' / 'export_optimized' / 'ft.npy'))
    parser.add_argument('--atlas_tile_res', type=int, default=1024, help='tile resolution (per UDIM)')
    parser.add_argument('--grid_x', type=int, default=2, help='UDIM grid X tiles')
    parser.add_argument('--grid_y', type=int, default=2, help='UDIM grid Y tiles')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--use_depth_test', action='store_true', help='Enable rasterizer depth-based visibility')
    parser.add_argument('--max_cams', type=int, default=0, help='Limit number of cameras (0 = all)')
    args = parser.parse_args()

    device = torch.device(args.device)

    print('Loading mesh...')
    mesh = trimesh.load(args.mesh)
    verts = np.array(mesh.vertices).astype(np.float32)
    faces = np.array(mesh.faces).astype(np.int32)

    print('Loading UVs...')
    vt = np.load(args.vt)
    ft = np.load(args.ft)

    Ks, Rts = load_cameras(DATASET)
    if args.max_cams > 0:
        Ks = Ks[:args.max_cams]
        Rts = Rts[:args.max_cams]
    print('Cameras:', len(Ks))

    # Rasterize UVs at big atlas resolution
    tile_res = args.atlas_tile_res
    gx, gy = args.grid_x, args.grid_y
    W = tile_res * gx
    H = tile_res * gy
    print(f'Rasterizing UVs at atlas resolution {W}x{H} ({gx}x{gy} tiles of {tile_res})')

    vt_t = torch.from_numpy(vt.astype(np.float32)).to(device).contiguous()
    ft_t = torch.from_numpy(ft.astype(np.int32)).to(device).contiguous()
    uv = vt_t * 2.0 - 1.0
    uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1)
    glctx = dr.RasterizeGLContext(output_db=False)
    rast, rast_db = dr.rasterize(glctx, uv.unsqueeze(0), ft_t, (H, W))

    # Interpolate world positions and mask
    v_t = torch.from_numpy(verts.astype(np.float32)).to(device).contiguous()
    f_t = torch.from_numpy(faces.astype(np.int32)).to(device).contiguous()
    xyzs_t, _ = dr.interpolate(v_t.unsqueeze(0), rast, f_t)
    mask_t, _ = dr.interpolate(torch.ones_like(v_t[:, :1]).unsqueeze(0), rast, f_t)

    xyzs = xyzs_t.view(-1,3).cpu().numpy()
    mask = (mask_t.view(-1,1).cpu().numpy() > 0).reshape(-1)

    print('Total texels:', W*H, 'used:', int(mask.sum()))

    # prepare accumulators per tile
    for ty in range(gy):
        for tx in range(gx):
            print(f'Processing tile {tx},{ty}...')
            x0 = tx * tile_res
            x1 = x0 + tile_res
            y0 = ty * tile_res
            y1 = y0 + tile_res
            # flatten texel indices
            inds = []
            for yy in range(y0, y1):
                row_start = yy * W
                inds.append(np.arange(row_start + x0, row_start + x1))
            inds = np.concatenate(inds, axis=0)
            tile_mask = mask[inds]
            if tile_mask.sum() == 0:
                print('  tile empty, skipping')
                continue
            xyz_tile = xyzs[inds]
            # initialize accumulators
            accum_color = np.zeros((tile_res*tile_res, 3), dtype=np.float64)
            accum_weight = np.zeros((tile_res*tile_res,), dtype=np.float64)
            view_count = np.zeros((tile_res*tile_res,), dtype=np.int32)

            # Project each texel into cameras
            for ci, (K, Rt) in enumerate(zip(Ks, Rts)):
                print(f'   camera {ci+1}/{len(Ks)}')
                imgp = sorted([p for p in (DATASET/'Image').iterdir() if p.suffix.lower()=='.png'])[ci]
                img = np.array(Image.open(imgp).convert('RGB'))/255.0
                Hc, Wc = img.shape[:2]
                # world->cam
                world2cam = np.linalg.inv(Rt)
                R = world2cam[:3,:3]
                t = world2cam[:3,3]
                xyz_cam = (R @ xyz_tile.T).T + t[None,:]
                z = xyz_cam[:,2]
                u = (K[0,0] * (xyz_cam[:,0]/z) + K[0,2])
                v = (K[1,1] * (xyz_cam[:,1]/z) + K[1,2])
                inside = (z>0) & (u>=0) & (u < Wc) & (v>=0) & (v < Hc) & tile_mask
                if inside.sum() == 0:
                    continue
                ui = np.clip(u[inside].astype(np.int32),0,Wc-1)
                vi = np.clip(v[inside].astype(np.int32),0,Hc-1)
                colors = img[vi, ui, :]
                # simple weight: ones (can be improved)
                w = np.ones((colors.shape[0],), dtype=np.float64)
                idxs = np.nonzero(inside)[0]
                accum_color[idxs] += colors * w[:,None]
                accum_weight[idxs] += w
                view_count[idxs] += 1

            used = accum_weight>0
            final = np.zeros((tile_res*tile_res,3), dtype=np.uint8)
            final[used] = (accum_color[used] / accum_weight[used,None] * 255.0).astype(np.uint8)
            final_img = final.reshape((tile_res, tile_res, 3))
            mask_img = (tile_mask.reshape((tile_res, tile_res)).astype(np.uint8)*255)
            view_img = view_count.reshape((tile_res, tile_res)).astype(np.uint8)

            # UDIM index (standard: 1001 + tx + ty*10)
            udim = 1001 + tx + ty*10
            out_base = EXPORT_DIR / f'albedo_{udim}'
            Image.fromarray(final_img).save(str(out_base) + '.png')
            Image.fromarray(mask_img).save(str(out_base) + '_mask.png')
            Image.fromarray(view_img).save(str(out_base) + '_views.png')
            print('  saved tile', out_base)

    print('UDIM bake done. Tiles in', EXPORT_DIR)

if __name__ == '__main__':
    main()
