#!/usr/bin/env python3
"""
Convert an existing .gltf + external resources (bin + images) into a single .glb file.

This script uses pygltflib to read the glTF JSON, embed the buffer and images as
bufferViews, and write a binary .glb.

Usage:
  python tools/gltf_to_glb.py --gltf outputs/.../model.gltf --out outputs/.../model.glb

Dependencies: pygltflib
"""

import argparse
import os
from pathlib import Path
from pygltflib import GLTF2, Buffer, BufferView

import json
import struct


def embed_resources(gltf_path: Path, out_glb: Path):
    base = gltf_path.parent
    gltf = GLTF2().load(str(gltf_path))

    # Read existing model.bin
    if gltf.buffers and gltf.buffers[0].uri:
        bin_path = base / gltf.buffers[0].uri
        with open(bin_path, 'rb') as f:
            bin_bytes = f.read()
    else:
        bin_bytes = b''

    # Gather image bytes and compute offsets
    image_bytes_list = []
    image_paths = []
    for img in list(gltf.images or []):
        if img.uri:
            img_path = base / img.uri
            image_paths.append(img_path)
            with open(img_path, 'rb') as f:
                image_bytes_list.append(f.read())

    # Compose final binary blob: model.bin followed by all image bytes in order
    new_blob = bin_bytes
    offsets = []
    for img_b in image_bytes_list:
        offsets.append(len(new_blob))
        new_blob += img_b

    # Ensure bufferViews list exists
    if gltf.bufferViews is None:
        gltf.bufferViews = []

    # Add bufferViews for images and set image.bufferView/mimeType
    img_idx = 0
    for i, img in enumerate(list(gltf.images or [])):
        if img.uri:
            off = offsets[img_idx]
            length = len(image_bytes_list[img_idx])
            bv = BufferView(buffer=0, byteOffset=off, byteLength=length)
            bv_index = len(gltf.bufferViews)
            gltf.bufferViews.append(bv)
            gltf.images[i].bufferView = bv_index
            suff = image_paths[img_idx].suffix.lower()
            if suff == '.png':
                gltf.images[i].mimeType = 'image/png'
            elif suff in ('.jpg', '.jpeg'):
                gltf.images[i].mimeType = 'image/jpeg'
            else:
                gltf.images[i].mimeType = 'application/octet-stream'
            gltf.images[i].uri = None
            img_idx += 1

    # Now build a JSON dict from original glTF file and update bufferViews/images/buffers
    gltf_json = json.loads(gltf.to_json())

    # Ensure bufferViews exist
    if 'bufferViews' not in gltf_json or gltf_json['bufferViews'] is None:
        gltf_json['bufferViews'] = []

    # Append new bufferViews for images (we already appended BufferView objects to gltf.bufferViews earlier,
    # but to avoid relying on pygltflib internals, we compute image entries here)
    # First, compute existing bufferViews count
    existing_bv_count = len(gltf_json['bufferViews'])
    # We will append bufferViews for images in the same order as image_bytes_list
    # Build a list of dicts for new bufferViews
    new_bvs = []
    for off, img_b in zip(offsets, image_bytes_list):
        new_bvs.append({
            'buffer': 0,
            'byteOffset': off,
            'byteLength': len(img_b)
        })
    gltf_json['bufferViews'].extend(new_bvs)

    # Update images entries to reference bufferViews and set mimeType; remove uri
    img_idx = 0
    for i, img in enumerate(gltf_json.get('images', []) or []):
        if 'uri' in img and img['uri'] is not None:
            img['bufferView'] = existing_bv_count + img_idx
            suff = Path(image_paths[img_idx]).suffix.lower()
            if suff == '.png':
                img['mimeType'] = 'image/png'
            elif suff in ('.jpg', '.jpeg'):
                img['mimeType'] = 'image/jpeg'
            else:
                img['mimeType'] = 'application/octet-stream'
            img.pop('uri', None)
            img_idx += 1

    # Update buffer length
    if 'buffers' not in gltf_json or gltf_json['buffers'] is None:
        gltf_json['buffers'] = [{}]
    gltf_json['buffers'][0]['byteLength'] = len(new_blob)
    gltf_json['buffers'][0].pop('uri', None)

    # Create JSON bytes (must be UTF-8) and pad to 4-byte alignment with spaces (0x20)
    json_bytes = json.dumps(gltf_json, separators=(',', ':')).encode('utf-8')
    json_padding = (4 - (len(json_bytes) % 4)) % 4
    json_bytes_padded = json_bytes + (b' ' * json_padding)

    # Pad binary blob to 4-byte alignment
    bin_padding = (4 - (len(new_blob) % 4)) % 4
    bin_bytes_padded = new_blob + (b'\x00' * bin_padding)

    # Build GLB header
    magic = 0x46546C67
    version = 2
    total_length = 12 + 8 + len(json_bytes_padded) + 8 + len(bin_bytes_padded)

    out_glb.parent.mkdir(parents=True, exist_ok=True)
    with open(out_glb, 'wb') as f:
        f.write(struct.pack('<I', magic))
        f.write(struct.pack('<I', version))
        f.write(struct.pack('<I', total_length))

        # JSON chunk
        f.write(struct.pack('<I', len(json_bytes_padded)))
        f.write(b'JSON')
        f.write(json_bytes_padded)

        # BIN chunk
        f.write(struct.pack('<I', len(bin_bytes_padded)))
        f.write(b'BIN\x00')
        f.write(bin_bytes_padded)

    print(f'[INFO] wrote {out_glb} (GLB manual pack)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gltf', required=True, help='path to .gltf file')
    parser.add_argument('--out', required=True, help='output .glb path')
    args = parser.parse_args()

    embed_resources(Path(args.gltf), Path(args.out))


if __name__ == '__main__':
    main()
