import trimesh
import numpy as np
from typing import List, Tuple, Optional

def load_glb_as_lines(file_path: str, target_scale: float = 20.0) -> Optional[List[Tuple[float, float, float]]]:
    """
    Loads a GLB/GLTF file, flattens it, normalizes its size/position, 
    and extracts its unique edges for wireframe rendering.
    
    Args:
        file_path: Path to the .glb file.
        target_scale: The desired maximum dimension (bounding box size) of the model.
        
    Returns:
        A flat list of vertices (start, end, start, end...) for GL_LINES.
    """
    try:
        # Load the mesh (Scene or Mesh)
        scene_or_mesh = trimesh.load(file_path, force='mesh') # force='mesh' tries to squash scenes
        
        # Ensure we have a single mesh
        if isinstance(scene_or_mesh, trimesh.Scene):
            # Concatenate all geometries into one mesh
            # This handles transform hierarchies automatically
            mesh = scene_or_mesh.dump(concatenate=True)
        else:
            mesh = scene_or_mesh

        if not hasattr(mesh, 'edges_unique'):
            print(f"Model {file_path} has no edges.")
            return None

        # --- Optimization: Normalization ---
        # 1. Center the mesh
        center = mesh.centroid
        mesh.vertices -= center
        
        # 2. Scale to target size
        # Get bounding box extents
        extents = mesh.extents # [x, y, z] sizes
        max_extent = np.max(extents)
        
        if max_extent > 0:
            scale_factor = target_scale / max_extent
            mesh.vertices *= scale_factor
            print(f"Auto-scaled model by {scale_factor:.4f}x to fit size {target_scale}")
        
        # --- Optimization: Vertex Count ---
        # If the model is absurdly dense (> 50k edges), we might want to warn or simplify
        # For now, we trust the user, but print stats
        print(f"Processing {len(mesh.edges_unique)} unique edges...")

        # Extract Geometry
        edges = mesh.edges_unique
        verts = mesh.vertices
        
        # Map indices to actual coordinates
        # shape (N, 2, 3) -> N lines, start/end, xyz
        line_segments = verts[edges]
        
        # Flatten to list of floats for VBO
        # We need a flat list of tuples? No, existing code expected list of tuples.
        # Let's keep strict compatibility with previous entities.py expectation if possible,
        # OR optimize by returning numpy array directly?
        # entities.py Mesh expects: `self.data = np.array(vertices, dtype=np.float32)`
        # If we return a numpy array of shape (N*2, 3), that works perfectly.
        
        # Reshape to (N*2, 3) -> flat list of vertices
        flat_verts = line_segments.reshape(-1, 3)
        
        # Convert to list of tuples to match strict type hint of entities.py Mesh __init__
        # (Though np.array handles list of lists too, explicit tuples are safer for the type hint)
        # Actually, let's just return the numpy array if Mesh accepts it?
        # entities.py: Mesh.__init__ takes vertices: List... then does np.array(vertices).
        # We can just return the list.
        
        return flat_verts.tolist()
        
    except Exception as e:
        print(f"Failed to load/optimize model {file_path}: {e}")
        return None

def load_glb_full(file_path: str, target_scale: float = 20.0) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Loads a GLB/GLTF file and returns data for solid rendering:
    (Vertices, Normals, Colors) as numpy arrays of shape (N*3, 3) or (N*3, 4).
    """
    try:
        scene_or_mesh = trimesh.load(file_path, force='mesh')
        
        if isinstance(scene_or_mesh, trimesh.Scene):
            mesh = scene_or_mesh.dump(concatenate=True)
        else:
            mesh = scene_or_mesh

        # --- Normalization ---
        center = mesh.centroid
        mesh.vertices -= center
        
        extents = mesh.extents
        max_extent = np.max(extents)
        if max_extent > 0:
            scale_factor = target_scale / max_extent
            mesh.vertices *= scale_factor

        # --- Extract Faces, Normals, Colors ---
        # Vertices for triangles
        verts = mesh.vertices[mesh.faces].reshape(-1, 3)
        # Normals for lighting
        normals = mesh.face_normals.repeat(3, axis=0) # Per vertex
        
        # Colors (check if available, else default to beige)
        if hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) > 0:
            # trimesh vertex_colors are per mesh vertex, we need per face vertex
            colors = mesh.visual.vertex_colors[mesh.faces].reshape(-1, 4).astype(np.float32) / 255.0
        else:
            # Default Macintosh Beige
            beige = [0.85, 0.8, 0.7, 1.0]
            colors = np.tile(beige, (len(verts), 1)).astype(np.float32)

        return verts, normals, colors
        
    except Exception as e:
        print(f"Failed to load full model {file_path}: {e}")
        return None