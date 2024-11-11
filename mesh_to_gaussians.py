import open3d as o3d
from initializerdefs import GaussiansDef
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def create_aligned_ellipsoid(normals):
    # Normalize the normal vectors
    normals = torch.from_numpy(normals).float()
    normals = torch.nn.functional.normalize(normals, dim=1)
    # print(normals[torch.norm(normals, dim=1) < 0.000001])

    z_axis = torch.tensor([0, 0, 1], dtype=torch.float32, device=normals.device)

    def compute_perpendicular_axis(v1, v2, eps=1e-8):
        # Compute the cross product of v1 and v2
        cross_product = torch.cross(v1, v2, dim=1)

        # Compute the norm of the cross product
        cross_product_norm = torch.norm(cross_product, dim=1, keepdim=True)

        # Check if the cross product is zero (i.e., v1 and v2 are parallel)
        parallel_mask = (cross_product_norm < eps).squeeze()

        # Initialize the perpendicular axis with the cross product
        perpendicular_axis = cross_product.clone()

        # For parallel vectors, set the perpendicular axis to a unit vector
        # that is perpendicular to v1 (and v2)
        perpendicular_axis[parallel_mask] = torch.tensor(
            [1, 0, 0], dtype=v1.dtype, device=v1.device
        )
        perpendicular_axis[parallel_mask & (torch.abs(v1[:, 0]) > eps)] = torch.tensor(
            [0, 1, 0], dtype=v1.dtype, device=v1.device
        )
        perpendicular_axis[
            parallel_mask & (torch.abs(v1[:, 0]) <= eps) & (torch.abs(v1[:, 1]) > eps)
        ] = torch.tensor([0, 0, 1], dtype=v1.dtype, device=v1.device)

        # Normalize the perpendicular axis
        perpendicular_axis = perpendicular_axis / (
            torch.norm(perpendicular_axis, dim=1, keepdim=True) + eps
        )

        return perpendicular_axis

    def vectors_to_quaternions(v1, v2, eps=1e-8):
        assert v1.shape == v2.shape
        assert v1.shape[1] == 3

        assert not torch.isnan(v1).any(), "v1 contains NaNs"
        assert not torch.isnan(v2).any(), "v2 contains NaNs"

        v1_norm = torch.norm(v1, dim=1, keepdim=True)
        v2_norm = torch.norm(v2, dim=1, keepdim=True)

        assert not torch.isnan(v1_norm).any(), "v1 norm contains NaNs"
        assert not torch.isnan(v2_norm).any(), "v2 norm contains NaNs"

        v1 = v1 / (v1_norm + eps)
        v2 = v2 / (v2_norm + eps)

        dot_product = torch.sum(v1 * v2, dim=1, keepdim=True)
        dot_product = torch.clamp(
            dot_product, min=-1.0, max=1.0
        )  # Clamp dot product to [-1, 1]
        cross_product = torch.cross(v1, v2, dim=1)

        assert not torch.isnan(dot_product).any(), "Dot product contains NaNs"
        assert not torch.isnan(cross_product).any(), "Cross product contains NaNs"

        # Handle anti-parallel case
        anti_parallel_mask = dot_product <= -1.0 + eps

        # Compute rotation angle
        rotation_angle = torch.acos(dot_product)

        assert not torch.isnan(rotation_angle).any(), "Rotation angle contains NaNs"

        # Set rotation axis to a vector perpendicular to v1 and v2 for anti-parallel vectors
        perpendicular_axis = compute_perpendicular_axis(v1, v2, eps)

        assert not torch.isnan(
            perpendicular_axis
        ).any(), "Perpendicular axis contains NaNs"

        rotation_axis = torch.where(
            anti_parallel_mask, perpendicular_axis, cross_product
        )

        assert not torch.isnan(
            rotation_axis
        ).any(), "Rotation axis contains NaNs before normalization"

        rotation_axis = rotation_axis / (
            torch.norm(rotation_axis, dim=1, keepdim=True) + eps
        )  # Normalize rotation axis

        assert not torch.isnan(
            rotation_axis
        ).any(), "Rotation axis contains NaNs after normalization"

        quaternions = torch.cat(
            [
                torch.cos(rotation_angle / 2),
                rotation_axis * torch.sin(rotation_angle / 2),
            ],
            dim=1,
        )

        assert not torch.isnan(
            quaternions
        ).any(), "Quaternions contain NaNs before normalization"

        quaternions = quaternions / (
            torch.norm(quaternions, dim=1, keepdim=True) + eps
        )  # Normalize quaternions

        assert not torch.isnan(
            quaternions
        ).any(), "Quaternions contain NaNs after normalization"

        assert (
            torch.abs(torch.norm(quaternions, dim=1) - 1) < 1e-6
        ).all(), "Quaternions must be normalized"

        return quaternions

    # Get the quaternions that align z-axis with normals
    alignment_quat = vectors_to_quaternions(z_axis.expand_as(normals), normals)
    return alignment_quat.cpu().numpy()


def batch_triangles_to_splats(triangles, normals):
    """
    Convert multiple triangles to Gaussian splat parameters in a vectorized way.
    
    Args:
        triangles: np.array of shape (N, 3, 3) representing N triangles, each with 3 vertices of xyz
        normals: np.array of shape (N, 3) representing triangle normals (normalized)
        
    Returns:
        dict containing:
            centers: np.array(N, 3) - center positions of splats
            scales: np.array(N, 3) - scales in each axis for each splat
            rotations: np.array(N, 4) - rotations as quaternions (x,y,z,w)
    """
    N = triangles.shape[0]
       
    # Calculate triangle centers (centroids)
    centers = np.mean(triangles, axis=1)  # Shape: (N, 3)
    
    # Calculate edges for each triangle
    edges = np.array([
        triangles[:, 1] - triangles[:, 0],  # first edge
        triangles[:, 2] - triangles[:, 1],  # second edge
        triangles[:, 0] - triangles[:, 2]   # third edge
    ])  # Shape: (3, N, 3)
    
    # Calculate edge lengths
    edge_lengths = np.linalg.norm(edges, axis=2)  # Shape: (3, N)
    
    # Find longest edges for scale calculation
    major_scales = np.max(edge_lengths, axis=0)  # Shape: (N,)
    
    # Calculate major axes (normalized longest edges)
     # Calculate major axes (normalized longest edges)
    longest_edge_idx = np.argmax(edge_lengths, axis=0)
    major_axes = np.zeros((N, 3))
    
    # Add epsilon for numerical stability
    eps = 1e-8
    
    # Safer major axes calculation
    for i in range(N):
        edge_length = edge_lengths[longest_edge_idx[i], i]
        if edge_length > eps:  # Check for degenerate edges
            major_axes[i] = edges[longest_edge_idx[i], i] / edge_length
        else:
            # Fallback for degenerate triangles
            major_axes[i] = np.array([1.0, 0.0, 0.0])
    
    # Calculate minor axes using cross product
    minor_axes = np.cross(normals, major_axes)
    minor_norms = np.linalg.norm(minor_axes, axis=1, keepdims=True)
    
    # Handle cases where cross product is zero
    valid_minor = (minor_norms > eps).flatten()
    minor_axes[valid_minor] = minor_axes[valid_minor] / minor_norms[valid_minor]
    
    # For invalid minor axes, find an alternative perpendicular vector
    invalid_minor = ~valid_minor.squeeze()
    if np.any(invalid_minor):
        # Create perpendicular vector using a different approach
        fallback = np.array([0.0, 1.0, 0.0])
        minor_axes[invalid_minor] = np.cross(normals[invalid_minor], fallback)
        minor_axes[invalid_minor] /= np.linalg.norm(minor_axes[invalid_minor], axis=1, keepdims=True)
    

    # Calculate widths by projecting edges onto minor axes
    widths = np.zeros(N)
    for i in range(3):
        widths = np.maximum(widths, np.abs(np.sum(edges[i] * minor_axes, axis=1)))
    
    # Create scales array
    scales = np.column_stack([
        major_scales / 2.0,  # Scale along major axis
        widths / 2.0,        # Scale along minor axis
        np.full(N, 0.0001)   # Very small scale along normal
    ])
    
    # Create rotation matrices and convert to quaternions
    rotations = create_aligned_ellipsoid(normals)
    
    return {
        'centers': centers,
        'scales': scales,
        'rotations': rotations
    }


## VIS helpers

def create_disc_mesh(resolution=4):
    """Create a unit circle mesh in the XY plane"""
    angles = np.linspace(0, 2*np.pi, resolution, endpoint=False)
    vertices = np.array([[np.cos(theta), np.sin(theta), 0] for theta in angles])
    vertices = np.vstack([vertices, [0, 0, 0]])  # Add center vertex
    
    # Create triangles by connecting adjacent points to center
    triangles = np.array([[i, (i+1)%resolution, resolution] 
                         for i in range(resolution)])
    
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh

def quaternion_to_matrix(q):
    """
    Convert quaternion [w,x,y,z] to 4x4 rotation matrix.
    """
    w, x, y, z = q
    
    # Precompute common products
    xx = x * x
    xy = x * y
    xz = x * z
    xw = x * w
    yy = y * y
    yz = y * z
    yw = y * w
    zz = z * z
    zw = z * w
    
    # Build 3x3 rotation matrix
    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - zw),     2*(xz + yw)],
        [    2*(xy + zw), 1 - 2*(xx + zz),     2*(yz - xw)],
        [    2*(xz - yw),     2*(yz + xw), 1 - 2*(xx + yy)]
    ])
    
    # Create 4x4 transformation matrix
    matrix = np.eye(4)
    matrix[:3, :3] = R
    return matrix



def splats_to_oriented_discs(centers, scales, rotations, resolution=16):
    """
    Convert Gaussian splats to oriented elliptical discs for visualization
    
    Args:
        centers: np.array(N, 3) - center positions
        scales: np.array(N, 3) - scales in each axis
        rotations: np.array(N, 4) - quaternions (x,y,z,w)
        resolution: int - number of points around disc circumference
    
    Returns:
        list of o3d.geometry.TriangleMesh representing the oriented elliptical discs
    """
    # Create base disc mesh
    base_disc = create_disc_mesh(resolution)
    
    # Convert to numpy for transformations
    disc_vertices = np.asarray(base_disc.vertices)
    disc_triangles = np.asarray(base_disc.triangles)
    
    discs = []
    for center, scale, quat in zip(centers, scales, rotations):
        # Create copy of base disc
        disc = o3d.geometry.TriangleMesh()
        disc.vertices = o3d.utility.Vector3dVector(disc_vertices.copy())
        disc.triangles = o3d.utility.Vector3iVector(disc_triangles.copy())
        
        # Scale the disc - use x and y scales for proper elliptical shape
        # z scale is kept very small to make it effectively 2D
        scaling_matrix = np.eye(4)
        scaling_matrix[0, 0] = scale[0]  # X scale
        scaling_matrix[1, 1] = scale[1]  # Y scale
        scaling_matrix[2, 2] = scale[2]  # Z scale (very small)
        disc.transform(scaling_matrix)
        
        # Rotate using quaternion
        
        rotation_matrix = quaternion_to_matrix(quat)
        disc.transform(rotation_matrix)
        
        # Translate to center
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = center
        disc.transform(translation_matrix)
        
        # Compute vertex normals for better rendering
        disc.compute_vertex_normals()
        disc.paint_uniform_color([0, 0, 1])  # Blue for splats

        discs.append(disc)
    
    return discs



   
