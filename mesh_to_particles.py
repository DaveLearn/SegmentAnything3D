import open3d as o3d
import numpy as np
import pymeshfix
from initializerdefs import ParticlesSpatialDef
from typing import List

def clean_and_watertight_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    # remove floating parts
    # Get connected components of the mesh
    #triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    #triangle_clusters = np.asarray(triangle_clusters)
    #cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    # Find the largest cluster
    #largest_cluster_idx = cluster_n_triangles.argmax()
    
    # Create a mask for triangles in the largest cluster
    #triangles_to_remove = triangle_clusters != largest_cluster_idx
    #mesh.remove_triangles_by_mask(triangles_to_remove)
    
    # Remove isolated vertices
    #mesh.remove_unreferenced_vertices()

    mesh = mesh.filter_smooth_laplacian(number_of_iterations=1)

    v = np.array(mesh.vertices)
    t = np.array(mesh.triangles)
    nv, nt = pymeshfix.clean_from_arrays(v,t)
    clean_mesh = o3d.geometry.TriangleMesh()
    clean_mesh.vertices = o3d.utility.Vector3dVector(nv)
    clean_mesh.triangles = o3d.utility.Vector3iVector(nt)

    return clean_mesh #.filter_smooth_laplacian(number_of_iterations=1)


# creates particles from mesh, the resulting shape is hollow

def match_mesh_with_particles(mesh: o3d.geometry.TriangleMesh, particle_radius: float) -> ParticlesSpatialDef:
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, particle_radius*2)
    
    xyz: List[np.ndarray] = []
    radius: List[float] = []
    
    for voxel in voxel_grid.get_voxels():
            v_xyz = np.array([
                float(voxel.grid_index[0] * voxel_grid.voxel_size + voxel_grid.origin[0]),
                float(voxel.grid_index[1] * voxel_grid.voxel_size + voxel_grid.origin[1]), 
                float(voxel.grid_index[2] * voxel_grid.voxel_size + voxel_grid.origin[2])
            ])
            xyz.append(v_xyz)
            radius.append(particle_radius)
        
    return ParticlesSpatialDef(
        xyz = np.array(xyz),
        radius = np.array(radius)
    )


# creates particles from mesh, the resulting shape is solid, uses a dense voxel grid
def fill_mesh_with_particles(mesh: o3d.geometry.TriangleMesh, particle_radius: float) -> ParticlesSpatialDef:
    # Create a dense voxel grid from mesh bounds
    mesh_min = np.asarray(mesh.get_min_bound())
    mesh_max = np.asarray(mesh.get_max_bound())
    voxel_size = particle_radius * 2
    
    # Calculate grid dimensions
    dims = np.ceil((mesh_max - mesh_min) / voxel_size).astype(int)
    print(f"Dims: {dims}")
    
    # Create grid points
    x = np.arange(0, dims[0]) * voxel_size + mesh_min[0]
    y = np.arange(0, dims[1]) * voxel_size + mesh_min[1] 
    z = np.arange(0, dims[2]) * voxel_size + mesh_min[2]
    
    xx, yy, zz = np.meshgrid(x, y, z)
    points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1).astype(np.float32)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    


    # Check which points are inside mesh
    inside = scene.compute_occupancy(points).numpy().astype(bool)
    # 1 is inside, 0 is outside, get the list of inside_points
    inside_points = points[inside]



    # Create particles for inside points
    return ParticlesSpatialDef(
        xyz = inside_points,
        radius = np.full(len(inside_points), particle_radius)
    )
