# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

try:
    import mitsuba
    try:
        mitsuba.set_variant('cuda_ad_rgb')
    except Exception:
        pass
except Exception:
    mitsuba = None
    print('[WARN] mitsuba not available; continuing without mitsuba')

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import imageio
import numpy as np
import torch
import open3d as o3d
import trimesh
import xatlas
import nvdiffrast.torch as dr
from PIL import Image
from argparse import ArgumentParser
from model.brdf import NGPBRDF
import base64
import io
try:
    from pygltflib import GLTF2, Buffer, BufferView, Accessor, Asset, Scene, Node, Mesh as GLTFMesh, Primitive, Material, Image as GLTFImage, Texture, PBRMetallicRoughness, TextureInfo
except Exception:
    GLTF2 = None

def main():
    parser = ArgumentParser()
    parser.add_argument('--mesh')
    parser.add_argument('--ckpt')
    parser.add_argument('--emitter_path')
    parser.add_argument('--dir_save')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--tex_res', type=int, default=2048)
    parser.add_argument('--chunk_size', type=int, default=160000)
    args = parser.parse_args()

    dir_save = args.dir_save
    os.makedirs(dir_save, exist_ok=True)

    # Load material network
    device = torch.device(args.device)
    # try load vslf.npz and checkpoint; if missing, fall back to a dummy BRDF for testing
    try:
        mask = torch.load(os.path.join(args.emitter_path,'vslf.npz'),map_location='cpu')
        voxel_min = mask['voxel_min']
        voxel_max = mask['voxel_max']
    except Exception as e:
        print(f'[WARN] failed to load vslf.npz from {args.emitter_path}: {e}. Using default bbox.')
        voxel_min = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32)
        voxel_max = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

    try:
        if args.ckpt and os.path.exists(args.ckpt):
            material_net = NGPBRDF(voxel_min,voxel_max)
            state = torch.load(args.ckpt, map_location='cpu')
            # support both lightning-style and plain state_dicts
            if isinstance(state, dict) and 'state_dict' in state:
                state_dict = state['state_dict']
            else:
                state_dict = state
            weight = {}
            for k,v in state_dict.items():
                if 'material.' in k:
                    weight[k.replace('material.','')]=v
            try:
                material_net.load_state_dict(weight)
                print(f'[INFO] loaded material network from {args.ckpt}')
            except Exception as e:
                print(f'[WARN] failed to load material weights from ckpt: {e}. Using dummy material network.')
                material_net = None
        else:
            material_net = None
    except Exception as e:
        print(f'[WARN] error while loading checkpoint: {e}. Using dummy material network.')
        material_net = None

    if material_net is None:
        # fallback dummy material network
        class DummyBRDF(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def to(self, device):
                return self
            def __call__(self, xyz):
                # xyz: [N,3]
                B = xyz.shape[0]
                albedo = torch.ones((B,3), dtype=torch.float32) * 0.8
                roughness = torch.ones((B,1), dtype=torch.float32) * 0.5
                metallic = torch.zeros((B,1), dtype=torch.float32)
                return {'albedo': albedo, 'roughness': roughness, 'metallic': metallic}
        material_net = DummyBRDF()
        material_net.to(device)
        print('[INFO] using DummyBRDF for material queries (constant albedo=0.8, roughness=0.5, metallic=0.0)')

    mesh  = trimesh.load(args.mesh)
    v_np = np.array(mesh.vertices).astype(np.float32)
    f_np = np.array(mesh.faces).astype(np.int32)

    # unwrap uvs
    print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')
    atlas = xatlas.Atlas()
    atlas.add_mesh(v_np, f_np)

    # xatlas to unwarp
    path_ft_np = os.path.join(dir_save, 'ft.npy')
    path_vt_np = os.path.join(dir_save, 'vt.npy')
    if os.path.exists(path_ft_np) and os.path.exists(path_vt_np):
        print(f'[INFO] found existing UVs, loading from {path_ft_np} and {path_vt_np}')
        ft_np = np.load(path_ft_np)
        vt_np = np.load(path_vt_np)
    else:
        chart_options = xatlas.ChartOptions()
        chart_options.max_iterations = 0  # disable merge_chart for faster unwrap...
        pack_options = xatlas.PackOptions()
        atlas.generate(chart_options=chart_options, pack_options=pack_options)
        _, ft_np, vt_np = atlas[0] # [N], [M, 3], [N, 2]
        np.save(path_ft_np, ft_np)
        np.save(path_vt_np, vt_np)
    print(f'[INFO] finished: xatlas unwraps UVs for mesh: v={v_np.shape} f={f_np.shape} vt={vt_np.shape} ft={ft_np.shape}')
    
    vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
    ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)
    # padding
    uv = vt * 2.0 - 1.0 # uvs to range [-1, 1]
    uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]

    glctx = dr.RasterizeGLContext(output_db=False)
    tex_res = args.tex_res
    h, w = tex_res, tex_res
    # rasterize 2d texture vertices to texture image
    rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), ft, (h, w)) # [1, h, w, 4]rast
    # interpolate to get the corresponding 3D location of each pixel
    v = torch.from_numpy(v_np).to(device)
    f = torch.from_numpy(f_np).to(device)
    xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f) # [1, h, w, 3]
    mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f) # [1, h, w, 1]
    vt = vt.cpu()
    ft = ft.cpu()
    uv = uv.cpu()
    # masked query
    xyzs = xyzs.view(-1, 3).cpu()
    mask = (mask > 0).view(-1).cpu()
        
    # masked query
    xyzs = xyzs.view(-1, 3).cpu()
    mask = (mask > 0).view(-1).cpu()

    mat_dim = 5
    feats = torch.zeros(h * w, mat_dim, dtype=torch.float32).cpu()
    if mask.any():
        with torch.no_grad():
            xyzs = xyzs[mask] # [M, 3]
            chunk_size = args.chunk_size
            # batched inference to avoid OOM
            all_feats = torch.zeros((xyzs.shape[0], mat_dim)).cpu()
            head = 0
            while head < xyzs.shape[0]:
                tail = min(head + chunk_size, xyzs.shape[0])
                with torch.cuda.amp.autocast(enabled=False):
                    slice_xyzs = xyzs[head:tail].clone().detach().cuda()
                    pred = material_net(slice_xyzs)
                    slice_mats = torch.cat([pred['albedo'], pred['roughness'], pred['metallic']], dim=-1)
                    all_feats[head:tail] = slice_mats.cpu().float()
                    slice_xyzs = slice_xyzs.cpu()
                    del slice_xyzs
                head += chunk_size
            feats[mask] = all_feats

    feats = feats.view(h, w, -1)
    mask = mask.view(h, w)
    # quantize [0.0, 1.0] to [0, 255]
    feats = feats.cpu().numpy()
    feats = (feats * 255).astype(np.uint8)
    mask = mask.cpu().numpy()

    albedo = np.zeros((h, w, 3), dtype=np.uint8)
    roughness_metallic = np.zeros((h, w, 3), dtype=np.uint8)
    albedo[:, :, :3] = feats[:, :, :3]
    roughness_metallic[:, :, :2] = feats[:, :, 3:]

    img_albedo = Image.fromarray(albedo)
    path_albedo = os.path.join(dir_save, 'albedo.png')
    img_albedo.save(path_albedo)
    # roughness_metallic currently stores: channel0=roughness, channel1=metallic
    # Build glTF metallicRoughness texture which expects: R = occlusion (we'll set 255), G = roughness, B = metallic
    metallic_roughness = np.zeros((h, w, 3), dtype=np.uint8)
    if roughness_metallic is not None:
        metallic_roughness[..., 0] = 255
        metallic_roughness[..., 1] = roughness_metallic[:, :, 0]
        metallic_roughness[..., 2] = roughness_metallic[:, :, 1]
    path_metallic_roughness = os.path.join(dir_save, 'metallicRoughness.png')
    img_mr = Image.fromarray(metallic_roughness)
    img_mr.save(path_metallic_roughness)
    print(f'[INFO] saved albedo to {path_albedo}, saved metallicRoughness to {path_metallic_roughness}')

    # Export a single-file glb (glTF binary) that embeds the mesh + textures in glTF PBR material
    def save_glb_with_atlas(out_glb, vertices, faces, vt_np, ft_np, albedo_path, mr_path):
        if GLTF2 is None:
            print('[WARN] pygltflib not available, skipping glb export. Install pygltflib to enable glb export.')
            return

        # Duplicate vertices per-face to support per-face-vertex UVs created by xatlas
        # faces: (F,3) indices into vertices
        # ft_np: (F,3) indices into vt_np
        F = faces.shape[0]
        new_positions = vertices[faces].reshape(-1, 3).astype(np.float32)
        new_texcoords = vt_np[ft_np].reshape(-1, 2).astype(np.float32)
        new_indices = np.arange(new_positions.shape[0], dtype=np.uint32).reshape(F, 3)

        # compute normals per-triangle then assign to each vertex
        tri_v0 = new_positions[new_indices[:, 0]]
        tri_v1 = new_positions[new_indices[:, 1]]
        tri_v2 = new_positions[new_indices[:, 2]]
        n = np.cross(tri_v1 - tri_v0, tri_v2 - tri_v0)
        # normalize
        n_len = np.linalg.norm(n, axis=1, keepdims=True)
        n_len[n_len == 0] = 1.0
        n = n / n_len
        new_normals = np.repeat(n, 3, axis=0).astype(np.float32)

        # Prepare binary buffer: positions, normals, texcoords, indices
        # Align to 4 bytes for each bufferView
        def pad_to4(b):
            pad = (4 - (len(b) % 4)) % 4
            if pad:
                b += b'\x00' * pad
            return b

        bin_parts = []
        # positions
        bin_positions = new_positions.tobytes()
        bin_parts.append(pad_to4(bin_positions))
        pos_offset = 0
        pos_len = len(bin_positions)

        # normals
        bin_normals = new_normals.tobytes()
        norm_offset = pos_offset + len(bin_parts[0])
        bin_parts.append(pad_to4(bin_normals))
        norm_len = len(bin_normals)

        # texcoords (vec2)
        bin_tex = new_texcoords.tobytes()
        tex_offset = norm_offset + len(bin_parts[1])
        bin_parts.append(pad_to4(bin_tex))
        tex_len = len(bin_tex)

        # indices (uint32)
        bin_idx = new_indices.astype(np.uint32).ravel().tobytes()
        idx_offset = tex_offset + len(bin_parts[2])
        bin_parts.append(pad_to4(bin_idx))
        idx_len = len(bin_idx)

        bin_blob = b"".join(bin_parts)

        gltf = GLTF2()
        gltf.asset = Asset(version="2.0")

        # Buffer
        gltf.buffers = [Buffer(byteLength=len(bin_blob))]

        # BufferViews
        buffer_views = []
        # positions
        buffer_views.append(BufferView(buffer=0, byteOffset=pos_offset, byteLength=pos_len))
        # normals
        buffer_views.append(BufferView(buffer=0, byteOffset=norm_offset, byteLength=norm_len))
        # texcoords
        buffer_views.append(BufferView(buffer=0, byteOffset=tex_offset, byteLength=tex_len))
        # indices
        buffer_views.append(BufferView(buffer=0, byteOffset=idx_offset, byteLength=idx_len))
        gltf.bufferViews = buffer_views

        # Accessors
        accessors = []
        # positions accessor
        pos_min = new_positions.min(axis=0).tolist()
        pos_max = new_positions.max(axis=0).tolist()
        accessors.append(Accessor(bufferView=0, byteOffset=0, componentType=5126, count=new_positions.shape[0], type="VEC3", min=pos_min, max=pos_max))
        # normals accessor
        accessors.append(Accessor(bufferView=1, byteOffset=0, componentType=5126, count=new_normals.shape[0], type="VEC3"))
        # texcoords accessor
        accessors.append(Accessor(bufferView=2, byteOffset=0, componentType=5126, count=new_texcoords.shape[0], type="VEC2"))
        # indices accessor
        accessors.append(Accessor(bufferView=3, byteOffset=0, componentType=5125, count=new_indices.size, type="SCALAR"))
        gltf.accessors = accessors

        # Images as data URIs
        def img_to_datauri(path):
            with open(path, 'rb') as f:
                b = f.read()
            ext = os.path.splitext(path)[1].lower().lstrip('.')
            mime = 'image/png' if ext in ['png'] else 'image/jpeg'
            b64 = base64.b64encode(b).decode('ascii')
            return f'data:{mime};base64,{b64}'

        img_alb_uri = img_to_datauri(albedo_path)
        img_mr_uri = img_to_datauri(mr_path)

        gltf.images = [GLTFImage(uri=img_alb_uri), GLTFImage(uri=img_mr_uri)]
        gltf.textures = [Texture(source=0), Texture(source=1)]

        # Material
        mat = Material(pbrMetallicRoughness=PBRMetallicRoughness(baseColorTexture=TextureInfo(index=0), metallicRoughnessTexture=TextureInfo(index=1)))
        gltf.materials = [mat]

        # Mesh & Primitive
        prim = Primitive(attributes={"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, indices=3, material=0)
        gltf_mesh = GLTFMesh(primitives=[prim])
        gltf.meshes = [gltf_mesh]

        # Node and scene
        node = Node(mesh=0)
        gltf.nodes = [node]
        gltf.scenes = [Scene(nodes=[0])]
        gltf.scene = 0

        # Attach binary buffer
        gltf.set_binary_blob(bin_blob)

        # Save binary glb
        gltf.save_binary(out_glb)
        print(f'[INFO] saved glb to {out_glb}')

    out_glb = os.path.join(dir_save, 'scene.glb')
    try:
        save_glb_with_atlas(out_glb, v_np, f_np, vt_np, ft_np, path_albedo, path_metallic_roughness)
    except Exception as e:
        print('[WARN] failed to export glb:', e)

if __name__ == '__main__':
    main()