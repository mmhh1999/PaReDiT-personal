#!/usr/bin/env python3
"""
Generate a glTF 2.0 model from runtime results (mesh + vt/ft UVs + baked textures).

This script DOES NOT read any previously generated OBJ that contains flipped/incorrect
UVs. Instead it uses the original mesh vertex positions/faces and the runtime UV arrays
(`vt.npy`, `ft.npy`) produced by the bake/export pipeline and the produced textures
(albedo.png, roughness.png, metallic.png) to generate a glTF (`model.gltf` + `model.bin`)
and copies textures into the output folder. The roughness and metallic single-channel
images are packed into a single ORM image where R = occlusion placeholder (255),
G = roughness, B = metallic (this matches the glTF metallicRoughness texture packing
convention where roughness is G and metallic is B; occlusion often uses R).

Usage example:
  python tools/generate_gltf_from_results.py \
      --mesh data/fipt/real/classroom/scene.obj \
      --vt outputs/fipt_real_classroom/export_optimized/vt.npy \
      --ft outputs/fipt_real_classroom/export_optimized/ft.npy \
      --textures-dir outputs/fipt_real_classroom/export_optimized \
      --out outputs/fipt_real_classroom/gltf

Dependencies: trimesh, numpy, pillow
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh


def pack_orm(roughness_path, metallic_path, out_path):
    """Create ORM image (R=occlusion placeholder 255, G=roughness, B=metallic).
    Both roughness and metallic are expected to be single-channel (L) images and
    will be resized to match if necessary.
    """
    im_r = None
    im_g = None
    im_b = None

    if not os.path.exists(roughness_path) or not os.path.exists(metallic_path):
        raise FileNotFoundError('roughness or metallic image not found')

    im_r = Image.new('L', Image.open(roughness_path).size, color=255)  # occlusion placeholder
    im_g = Image.open(roughness_path).convert('L')
    im_b = Image.open(metallic_path).convert('L')

    # Resize to the same size if needed
    if im_g.size != im_b.size:
        im_b = im_b.resize(im_g.size, Image.BILINEAR)

    orm = Image.merge('RGB', (im_r, im_g, im_b))
    orm.save(out_path)
    return out_path


def build_gltf(out_dir, positions, normals, texcoords, indices, basecolor_name, orm_name):
    """Write a .gltf + .bin pair into out_dir. Images must be present in out_dir.
    This writes a simple scene with one node/mesh/primitive and a PBR material using
    baseColorTexture and metallicRoughnessTexture.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare binary buffer (concatenate attributes and indices)
    # Ensure 4-byte alignment for each bufferView
    def pad(b):
        pad_len = (4 - (len(b) % 4)) % 4
        if pad_len:
            return b + (b'\x00' * pad_len)
        return b

    pos_bytes = positions.astype(np.float32).tobytes()
    norm_bytes = normals.astype(np.float32).tobytes()
    tex_bytes = texcoords.astype(np.float32).tobytes()
    idx_bytes = indices.astype(np.uint32).tobytes()

    # compute offsets with padding
    offset = 0
    pos_b = pad(pos_bytes)
    offset_pos = offset
    offset += len(pos_b)

    norm_b = pad(norm_bytes)
    offset_norm = offset
    offset += len(norm_b)

    tex_b = pad(tex_bytes)
    offset_tex = offset
    offset += len(tex_b)

    idx_b = pad(idx_bytes)
    offset_idx = offset
    offset += len(idx_b)

    total_len = offset

    buffer_filename = 'model.bin'
    buffer_path = out_dir / buffer_filename
    with open(buffer_path, 'wb') as f:
        f.write(pos_b)
        f.write(norm_b)
        f.write(tex_b)
        f.write(idx_b)

    # accessors
    accessor_count_pos = positions.shape[0]
    accessor_count_idx = indices.shape[0]

    # compute min/max for positions
    pos_min = positions.min(axis=0).tolist()
    pos_max = positions.max(axis=0).tolist()

    gltf = {
        'asset': {'version': '2.0'},
        'buffers': [
            {'uri': buffer_filename, 'byteLength': total_len}
        ],
        'bufferViews': [
            {'buffer': 0, 'byteOffset': offset_pos, 'byteLength': len(pos_b)},
            {'buffer': 0, 'byteOffset': offset_norm, 'byteLength': len(norm_b)},
            {'buffer': 0, 'byteOffset': offset_tex, 'byteLength': len(tex_b)},
            {'buffer': 0, 'byteOffset': offset_idx, 'byteLength': len(idx_b)},
        ],
        'accessors': [
            # POSITION
            {
                'bufferView': 0,
                'byteOffset': 0,
                'componentType': 5126,  # FLOAT
                'count': accessor_count_pos,
                'type': 'VEC3',
                'min': pos_min,
                'max': pos_max,
            },
            # NORMAL
            {
                'bufferView': 1,
                'byteOffset': 0,
                'componentType': 5126,
                'count': accessor_count_pos,
                'type': 'VEC3',
            },
            # TEXCOORD_0
            {
                'bufferView': 2,
                'byteOffset': 0,
                'componentType': 5126,
                'count': accessor_count_pos,
                'type': 'VEC2',
            },
            # INDICES
            {
                'bufferView': 3,
                'byteOffset': 0,
                'componentType': 5125,  # UNSIGNED_INT
                'count': accessor_count_idx,
                'type': 'SCALAR',
            },
        ],
        'images': [
            {'uri': os.path.basename(basecolor_name)},
            {'uri': os.path.basename(orm_name)},
        ],
        'textures': [
            {'source': 0},
            {'source': 1},
        ],
        'materials': [
            {
                'pbrMetallicRoughness': {
                    'baseColorTexture': {'index': 0},
                    'metallicRoughnessTexture': {'index': 1},
                },
                'name': 'pbr_material'
            }
        ],
        'meshes': [
            {
                'primitives': [
                    {
                        'attributes': {
                            'POSITION': 0,
                            'NORMAL': 1,
                            'TEXCOORD_0': 2,
                        },
                        'indices': 3,
                        'material': 0,
                    }
                ]
            }
        ],
        'nodes': [
            {'mesh': 0, 'name': 'node_0'}
        ],
        'scenes': [
            {'nodes': [0]}
        ],
        'scene': 0,
    }

    gltf_path = out_dir / 'model.gltf'
    with open(gltf_path, 'w') as f:
        json.dump(gltf, f, indent=2)

    print(f'[INFO] wrote {gltf_path} and {buffer_path} (textures must be in the same folder)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', required=True, help='original mesh (OBJ/PLY) path')
    parser.add_argument('--vt', required=True, help='vt.npy (UV vertices)')
    parser.add_argument('--ft', required=True, help='ft.npy (face-to-uv indices)')
    parser.add_argument('--textures-dir', required=True, help='directory containing albedo.png, roughness.png, metallic.png')
    parser.add_argument('--out', required=True, help='output directory for glTF')
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    vt_path = Path(args.vt)
    ft_path = Path(args.ft)
    tex_dir = Path(args.textures_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load source mesh
    mesh = trimesh.load(mesh_path)
    v_np = np.array(mesh.vertices)
    f_np = np.array(mesh.faces)

    # Load UVs produced by the pipeline
    vt = np.load(vt_path)  # [M,2]
    ft = np.load(ft_path).astype(np.int64)  # [F,3]

    # Build expanded vertex arrays by duplicating vertices per face-uv pair
    positions = []
    normals = []
    texcoords = []
    indices = []

    # Use original per-vertex normals if available
    if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
        vnorms = np.array(mesh.vertex_normals)
    else:
        vnorms = None

    cur_index = 0
    for face_i in range(f_np.shape[0]):
        face = f_np[face_i]
        uv_idx = ft[face_i]
        for j in range(3):
            pos = v_np[face[j]]
            positions.append(pos)
            if vnorms is not None:
                normals.append(vnorms[face[j]])
            else:
                normals.append([0.0, 0.0, 0.0])
            texcoords.append(vt[uv_idx[j]])
            indices.append(cur_index)
            cur_index += 1

    positions = np.asarray(positions, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    texcoords = np.asarray(texcoords, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.uint32)

    # Load textures
    albedo_path = tex_dir / 'albedo.png'
    roughness_path = tex_dir / 'roughness.png'
    metallic_path = tex_dir / 'metallic.png'

    if not albedo_path.exists():
        raise FileNotFoundError(f'albedo not found: {albedo_path}')
    if not roughness_path.exists() or not metallic_path.exists():
        raise FileNotFoundError('roughness or metallic not found in textures dir')

    # pack ORM and copy textures into out_dir
    orm_out = out_dir / 'orm.png'
    pack_orm(str(roughness_path), str(metallic_path), str(orm_out))

    # copy albedo
    from shutil import copy2
    copy2(str(albedo_path), str(out_dir / 'albedo.png'))

    # Build glTF and binary
    build_gltf(out_dir, positions, normals, texcoords, indices, str(out_dir / 'albedo.png'), str(orm_out))


if __name__ == '__main__':
    main()
