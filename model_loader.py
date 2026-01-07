import trimesh
import numpy as np
from typing import List, Tuple, Optional

def load_glb_as_lines(file_path: str, scale: float = 1.0) -> Optional[List[Tuple[float, float, float]]]:
    """
    Loads a GLB/GLTF file and extracts its edges for wireframe rendering.
    Returns a flat list of vertices (start, end, start, end...) for GL_LINES.
    """
    try:
        # Load the mesh
        # trimesh.load can return a Scene or a Mesh. We handle both.
        scene_or_mesh = trimesh.load(file_path)
        
        meshes = []
        if isinstance(scene_or_mesh, trimesh.Scene):
            # Aggregate all meshes in the scene
            for geometry in scene_or_mesh.geometry.values():
                meshes.append(geometry)
        else:
            meshes.append(scene_or_mesh)
            
        all_lines = []
        
        for mesh in meshes:
            if not hasattr(mesh, 'edges_unique'):
                continue
                
            # Get unique edges (pairs of vertex indices)
            edges = mesh.edges_unique
            # Get vertices
            verts = mesh.vertices
            
            # Map indices to actual coordinates
            # shape (N, 2, 3) -> N lines, start/end, xyz
            line_segments = verts[edges]
            
            # Apply scale
            line_segments *= scale
            
            # Center the model? Optional. Let's not auto-center for now, assume origin is correct.
            # But usually GLB origins are at feet.
            
            # Flatten to list of tuples
            for start, end in line_segments:
                all_lines.append((start[0], start[1], start[2]))
                all_lines.append((end[0], end[1], end[2]))
                
        print(f"Loaded {file_path}: {len(all_lines)//2} edges.")
        return all_lines
        
    except Exception as e:
        print(f"Failed to load model {file_path}: {e}")
        return None
