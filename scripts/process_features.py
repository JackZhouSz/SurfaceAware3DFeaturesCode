import os
import re
import sys
import argparse
import numpy as np
from tqdm import tqdm
import torch
import trimesh
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import get_feature_network, compute_features, get_dummy_args

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Diff3D'))
from Diff3D.diffusion import init_pipe
from Diff3D.dino import init_dino
from Diff3D.dataloaders.mesh_container import MeshContainer

def get_model(exp_path, args, device):
    with torch.no_grad():
        model, _, _, _ = get_feature_network(args, os.path.join(exp_path, 'feature_network.pt'))
        model = model.to(device)
    return model

def find_obj_files(source_path):
    obj_files = []
    for root, _, files in os.walk(source_path):
        for f in sorted(files):
            if f.endswith('.obj'):
                obj_files.append(os.path.join(root, f))
    return obj_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Folder containing .obj files (searched recursively)")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for feature extraction. Diff3F uses 'naked human' for humans as prompt.")
    parser.add_argument("--ignore", type=str, default=None, help="Ignore .obj files whose name contains this substring")
    parser.add_argument("--num_views", type=int, default=100, help="Number of views used. Only perfect square values (4, 9, 16, 25, ...) will use the full requested number of views")
    parser.add_argument("--range", type=str, default=None, help="Range of meshes to process after sorting, e.g. '0-50' or '50-100'")
    parser.add_argument("--target", type=str, default=None, help="Output folder for features. Defaults to <source>_features")

    cli_args = parser.parse_args()

    device = 'cuda'
    PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')

    # Load both SAF3D models
    args = get_dummy_args()
    model_quadruped = get_model(os.path.join(PROJECT_ROOT, 'experiments/humans_animals'), args, device)
    model_all = get_model(os.path.join(PROJECT_ROOT, 'experiments/all_shapes'), args, device)

    # Load diffusion + dino models
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    pipe = init_pipe(device)
    dino_model = init_dino(device)

    def natural_sort_key(s):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

    obj_files = sorted(find_obj_files(cli_args.source), key=natural_sort_key)
    if cli_args.ignore:
        obj_files = [f for f in obj_files if cli_args.ignore not in os.path.basename(f)]
    if cli_args.range:
        start, end = map(int, cli_args.range.split('-'))
        obj_files = obj_files[start:end]
    target = cli_args.target if cli_args.target else cli_args.source.rstrip('/') + '_features'
    raw_dir = os.path.join(target, 'raw')
    quadruped_dir = os.path.join(target, 'quadruped')
    all_dir = os.path.join(target, 'all')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(quadruped_dir, exist_ok=True)
    os.makedirs(all_dir, exist_ok=True)
    print(f"Found {len(obj_files)} .obj files, saving to {target}")

    with torch.no_grad():
        for obj_path in tqdm(obj_files):
            obj_name = os.path.splitext(os.path.basename(obj_path))[0]

            raw_path = os.path.join(raw_dir, f"{obj_name}.pt")
            quadruped_path = os.path.join(quadruped_dir, f"{obj_name}.pt")
            all_path = os.path.join(all_dir, f"{obj_name}.pt")

            mesh = trimesh.load_mesh(obj_path, process=False, skip_materials=True)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

            source_mesh = MeshContainer(np.array(mesh.vertices), np.array(mesh.faces))
            vertex_features = compute_features(device, pipe, dino_model, source_mesh, prompt=cli_args.prompt, num_views=cli_args.num_views)
            torch.save(vertex_features.cpu(), raw_path)

            encoded_quadruped, _ = model_quadruped(vertex_features.to(device=device, dtype=torch.float32))
            encoded_all, _ = model_all(vertex_features.to(device=device, dtype=torch.float32))
            torch.save(encoded_quadruped.cpu(), quadruped_path)
            torch.save(encoded_all.cpu(), all_path)

            tqdm.write(f"Saved {raw_path}, {quadruped_path}, {all_path}")
