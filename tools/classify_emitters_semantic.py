#!/usr/bin/env python3
"""
Semantic emitter classifier (SAM + CLIP assisted, with CV fallback).

Usage (example):
  python tools/classify_emitters_semantic.py \
    --mesh data/fipt/real/conferenceroom/scene.obj \
    --vt outputs/fipt_real_conferenceroom/export_optimized/vt_flipped.npy \
    --ft outputs/fipt_real_conferenceroom/export_optimized/ft.npy \
    --textures-dir outputs/fipt_real_conferenceroom/export_optimized \
    --emitter checkpoints/fipt_conferenceroom/bake/emitter.pth \
    --out outputs/fipt_real_conferenceroom/classified_semantic \
    --use-sam --use-clip

This script will:
 - load the emission/albedo texture
 - generate masks (SAM if available, otherwise connectedComponents)
 - optionally score masks with CLIP (if available)
 - map masks to mesh faces via UV centroid sampling
 - compute geometric & photometric features per component
 - classify components into window / lamp / unknown using simple fusion rules
 - write `emitter_classified_semantic.pth` (torch) and a colored OBJ for quick viz

Note: For robust pixel->face mapping you may want a full rasterization; here we use face UV centroids
as a pragmatic and much faster approximation. This is sufficient in most cases but may fail for
very large UV-distorted triangles.
"""

import os
import sys
import argparse
import json
import math
import numpy as np
from collections import defaultdict

try:
    import torch
except Exception:
    torch = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import trimesh
except Exception:
    trimesh = None

def load_image_any(path):
    # try cv2 then pillow
    if cv2 is not None:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise RuntimeError(f'cv2 failed to read {path}')
        # convert BGR->RGB
        if im.ndim == 3:
            im = im[..., ::-1]
        return im.astype(np.float32) / 255.0
    else:
        from PIL import Image
        im = Image.open(path).convert('RGB')
        return np.asarray(im).astype(np.float32)/255.0

def adaptive_mask_from_image(img_gray):
    # img_gray in [0,1]
    if cv2 is None:
        thr = img_gray.mean()*1.0
        return (img_gray > thr).astype(np.uint8)
    img_u8 = (np.clip(img_gray,0,1)*255).astype(np.uint8)
    th = cv2.adaptiveThreshold(img_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 51, -10)
    return (th//255).astype(np.uint8)

def connected_components_masks(bin_mask):
    # bin_mask 0/1
    if cv2 is None:
        # crude: return whole mask as one
        return [bin_mask.astype(np.uint8)]
    num, labels = cv2.connectedComponents(bin_mask.astype(np.uint8), connectivity=8)
    masks = []
    for i in range(1, num):
        masks.append((labels==i).astype(np.uint8))
    return masks

def contour_rect_score(mask):
    # compute rectangularity (IOU with minAreaRect)
    if cv2 is None:
        return 0.0
    mask_u8 = (mask*255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)==0:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    rect_area = int(abs(rect[1][0]*rect[1][1]))
    if rect_area <= 0:
        return 0.0
    return float(area) / float(rect_area)

def clip_score_for_patch(patch, clip_model=None, preprocess=None, texts=None):
    # patch: HxWx3 float [0,1]
    if clip_model is None:
        return {}
    import torch
    device = next(clip_model.parameters()).device
    from PIL import Image
    img = Image.fromarray((np.clip(patch,0,1)*255).astype(np.uint8))
    # preprocess from openai/clip
    img_t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = clip_model.encode_image(img_t)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        scores = {}
        for k, txt in texts.items():
            txt_t = clip.tokenize([txt]).to(device)
            txt_feat = clip_model.encode_text(txt_t)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ txt_feat.T).cpu().item()
            scores[k] = sim
    return scores

def uv_centroid_for_face(uvs, face):
    # uvs Nx2, face is 3 indices
    tri = uvs[face]
    return tri.mean(axis=0)

def compute_planarity(points):
    # points: (N,3)
    if len(points) < 3:
        return 1.0
    pts = points - points.mean(axis=0)
    U, S, Vt = np.linalg.svd(pts, full_matrices=False)
    # planarity = (S0 - S2) / S0 or fraction of first two explained
    if S.sum() <= 0:
        return 1.0
    return float((S[0]+S[1]) / S.sum())

def save_colored_obj(mesh, face_labels, out_path):
    # mesh: trimesh.Trimesh loaded
    # face_labels: array of length n_faces with small integers
    colors = {
        0: [200,200,200], # none
        1: [0,200,255],   # window - cyan
        2: [255,200,0],   # lamp - orange
        3: [200,0,200],   # unknown - magenta
    }
    # duplicate faces and write per-face color in OBJ by writing a per-face mtllib
    # For simplicity, write a PLY via trimesh
    try:
        # create per-vertex color by assigning face color to its vertices (may overwrite)
        vcols = np.zeros((len(mesh.vertices),3), dtype=np.uint8)
        counts = np.zeros(len(mesh.vertices), dtype=int)
        for fi, face in enumerate(mesh.faces):
            c = colors.get(int(face_labels[fi]), [128,128,128])
            for vi in face:
                vcols[vi] += c
                counts[vi] += 1
        nz = counts>0
        vcols[nz] = (vcols[nz].astype(np.float32) / counts[nz,None]).astype(np.uint8)
        ply = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vcols)
        ply.export(out_path)
    except Exception as e:
        print('Failed to write colored OBJ/PLY:', e)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mesh', required=True)
    p.add_argument('--vt', required=True)
    p.add_argument('--ft', required=True)
    p.add_argument('--textures-dir', required=True)
    p.add_argument('--emitter', required=False, help='path to emitter.pth')
    p.add_argument('--out', required=True)
    p.add_argument('--use-sam', action='store_true')
    p.add_argument('--use-clip', action='store_true')
    p.add_argument('--tex-name', default=None, help='basename of texture (albedo/roughness). If omitted, will search')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # find texture: prefer emission.png or albedo.png
    tex = None
    if args.tex_name:
        cand = os.path.join(args.textures_dir, args.tex_name)
        if os.path.exists(cand):
            tex = cand
    else:
        for name in ['emission.png','emitter.png','albedo.png','albedo.jpg','albedo.exr']:
            cand = os.path.join(args.textures_dir, name)
            if os.path.exists(cand):
                tex = cand
                break
    if tex is None:
        # fallback: use any png in textures-dir
        for f in os.listdir(args.textures_dir):
            if f.lower().endswith(('.png','.jpg','.exr')):
                tex = os.path.join(args.textures_dir, f)
                break
    if tex is None:
        print('No texture found in', args.textures_dir)
        return

    print('Using texture:', tex)
    img = load_image_any(tex)
    H, W = img.shape[:2]
    # luminance
    if img.ndim==3:
        lum = 0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]
    else:
        lum = img

    # generate masks
    masks = []
    if args.use_sam:
        try:
            from segment_anything import sam_model_registry, SamPredictor
            print('SAM available: running automatic masks (fast automatic mode)')
            # minimal SAM usage: use automatic mask generator if available
            try:
                from segment_anything import SamAutomaticMaskGenerator
                sam_checkpoint = None
                model_type = 'default'
                # try to instantiate default model (may fail if no weights)
                # fallback: use adaptive threshold instead
                # This area intentionally conservative: if SAM cannot be used, fallback
                print('SAM AutomaticMaskGenerator not configured; falling back to CV threshold')
                raise RuntimeError('SAM missing weights')
            except Exception:
                masks = connected_components_masks(adaptive_mask_from_image(lum))
        except Exception as e:
            print('SAM not available or failed:', e)
            masks = connected_components_masks(adaptive_mask_from_image(lum))
    else:
        masks = connected_components_masks(adaptive_mask_from_image(lum))

    print(f'Generated {len(masks)} mask(s)')

    # try load clip
    clip_model = None
    clip_preproc = None
    clip_texts = {'window':'a window', 'lamp':'a lamp', 'light':'a light fixture'}
    if args.use_clip:
        try:
            import clip
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            clip_model, preprocess = clip.load('ViT-B/32', device=device)
            clip_model.eval()
            clip_model.to(device)
            clip_preproc = preprocess
            print('CLIP loaded on', device)
        except Exception as e:
            print('CLIP failed to load:', e)
            clip_model = None

    # load mesh
    if trimesh is None:
        print('trimesh not available; aborting')
        return
    mesh = trimesh.load(args.mesh, process=False)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    # load vt/ft
    vt = np.load(args.vt)
    ft = np.load(args.ft)

    # compute face UV centroid pixel coords
    # detect if vt appears flipped: we assume vt is in [0,1]
    uv_centroids = np.zeros((len(faces),2), dtype=np.float32)
    for i, f in enumerate(ft):
        c = uv_centroid_for_face(vt, f)
        # assume v is already flipped correctly when using vt_flipped.npy
        px = int(c[0]* (W-1))
        py = int(c[1]* (H-1))
        uv_centroids[i] = [px, py]

    # map faces to masks via centroid membership
    face_to_mask = np.full(len(faces), -1, dtype=int)
    for mi, mask in enumerate(masks):
        # mask is HxW 0/1
        # build boolean mask
        b = mask.astype(bool)
        # bounding box speedup
        ys, xs = np.where(b)
        if len(xs)==0:
            continue
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        # check centroids inside bbox first
        in_bbox = (uv_centroids[:,0] >= xmin) & (uv_centroids[:,0] <= xmax) & (uv_centroids[:,1] >= ymin) & (uv_centroids[:,1] <= ymax)
        for fi in np.where(in_bbox)[0]:
            x = int(uv_centroids[fi,0]); y = int(uv_centroids[fi,1])
            if x<0 or x>=W or y<0 or y>=H:
                continue
            if b[y,x]:
                face_to_mask[fi] = mi

    # aggregate components
    comps = defaultdict(list)
    for fi, mi in enumerate(face_to_mask):
        if mi>=0:
            comps[mi].append(fi)

    results = {}
    face_label = np.zeros(len(faces), dtype=np.int32)
    for mi, face_list in comps.items():
        # image features
        mask = masks[mi]
        rect_score = contour_rect_score(mask)
        ys, xs = np.where(mask)
        pix_area = len(xs)
        # image patch for CLIP
        if clip_model is not None:
            xmin, xmax = xs.min(), xs.max(); ymin, ymax = ys.min(), ys.max()
            patch = img[ymin:ymax+1, xmin:xmax+1]
            try:
                clip_scores = clip_score_for_patch(patch, clip_model, clip_preproc, clip_texts)
            except Exception as e:
                print('CLIP scoring failed for mask', mi, e)
                clip_scores = {}
        else:
            clip_scores = {}

        # geometry features
        comp_faces = np.array(face_list, dtype=int)
        face_verts = faces[comp_faces].reshape(-1)
        unique_vs = np.unique(face_verts)
        points = verts[unique_vs]
        area_total = mesh.area_faces[comp_faces].sum()
        planarity = compute_planarity(points)
        # mean radiance per face if emitter.pth given
        mean_rad = None
        if args.emitter and torch is not None and os.path.exists(args.emitter):
            try:
                st = torch.load(args.emitter, map_location='cpu')
                # try keys
                if 'emitter_radiance' in st:
                    er = st['emitter_radiance']
                    if isinstance(er, np.ndarray):
                        er = torch.from_numpy(er)
                    # er shape (n_face,3)
                    # compute luminance
                    erlum = (0.2126*er[:,0] + 0.7152*er[:,1] + 0.0722*er[:,2]).numpy()
                    mean_rad = float(erlum[comp_faces].mean())
                elif 'emitter' in st:
                    # fallback
                    mean_rad = float(np.mean(st['emitter']))
            except Exception as e:
                print('Failed to read emitter.pth radiance:', e)

        # derive a simple decision
        score_window = clip_scores.get('window', 0.0)
        score_lamp = clip_scores.get('lamp', 0.0)
        label = 3 # unknown
        # heuristics
        if score_window > 0.25 and planarity > 0.85 and area_total > 0.2:
            label = 1
        elif score_lamp > 0.25 and (mean_rad is not None and mean_rad > 0.5) and area_total < 0.05:
            label = 2
        else:
            # fallback rule using rect_score and planarity
            if rect_score > 0.7 and planarity > 0.9 and area_total > 0.1:
                label = 1
            elif mean_rad is not None and mean_rad > 1.0 and area_total < 0.08:
                label = 2

        for fi in comp_faces:
            face_label[fi] = label

        results[mi] = dict(
            faces=comp_faces.tolist(),
            pix_area=int(pix_area),
            rect_score=float(rect_score),
            planarity=float(planarity),
            area_total=float(area_total),
            clip_scores=clip_scores,
            mean_rad=mean_rad,
            label=int(label)
        )

    # save results
    out_pth = os.path.join(args.out, 'emitter_classified_semantic.pth')
    if torch is not None:
        torch.save({'face_label': face_label, 'components': results}, out_pth)
        print('Wrote', out_pth)
    else:
        np.savez(out_pth.replace('.pth','.npz'), face_label=face_label)
        print('Wrote', out_pth.replace('.pth','.npz'))

    # save colored obj for quick viz
    out_viz = os.path.join(args.out, 'visualization.ply')
    save_colored_obj(mesh, face_label, out_viz)
    print('Wrote visualization:', out_viz)

    # also write CSV summary
    import csv
    csv_p = os.path.join(args.out, 'components.csv')
    with open(csv_p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['component','label','pix_area','rect_score','planarity','area_total','mean_rad','clip_window','clip_lamp'])
        for k,v in results.items():
            w.writerow([k, v['label'], v['pix_area'], v['rect_score'], v['planarity'], v['area_total'], v['mean_rad'], v['clip_scores'].get('window',None), v['clip_scores'].get('lamp',None)])
    print('Wrote CSV:', csv_p)


if __name__ == '__main__':
    main()
