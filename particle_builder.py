from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
from psdstaticdataset import StaticDataset
from psdframe import Frame
from initializerdefs import SceneSetup, ObjectsSpatialDef, ParticlesSpatialDef,ObjectSpatialDef, GaussiansDef
from PIL import Image
from mesh_to_gaussians import mesh_to_gaussians
import sam3d
from download_sam import get_sam_checkpoint
from segment_anything import build_sam, SamAutomaticMaskGenerator
from util import Voxelize, num_to_natural
import numpy as np
import open3d as o3d
import torch
import pointops
from mesh_to_particles import clean_and_watertight_mesh, fill_mesh_with_particles

import logging
logger = logging.getLogger("sam3d-builder")

logger.info("Downloading/Locating SAM checkpoint...")
sam_checkpoint = get_sam_checkpoint()
logger.info(f"Sam checkpoint in located at {sam_checkpoint}")

VOXEL_SIZE = 0.0035
TH = 50
PARTICLE_RADIUS = 0.007

class LabelledPcd(TypedDict):
    coord: np.ndarray
    color: np.ndarray
    group: np.ndarray
    normals: np.ndarray


def normals_from_depth(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    device = depth.device
    dtype = depth.dtype

    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create grid of pixel coordinates
    x = torch.arange(0, W, device=device, dtype=dtype)
    y = torch.arange(0, H, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="xy")

    # Compute 3D coordinates
    X = (xx - cx) * depth / fx
    Y = (yy - cy) * depth / fy
    Z = depth

    # Compute vectors from neighboring pixels
    Vx = torch.zeros((H, W, 3), device=device, dtype=dtype)
    Vy = torch.zeros((H, W, 3), device=device, dtype=dtype)

    Vx[:, :-1, 0] = X[:, 1:] - X[:, :-1]
    Vx[:, :-1, 1] = Y[:, 1:] - Y[:, :-1]
    Vx[:, :-1, 2] = Z[:, 1:] - Z[:, :-1]

    Vy[:-1, :, 0] = X[1:, :] - X[:-1, :]
    Vy[:-1, :, 1] = Y[1:, :] - Y[:-1, :]
    Vy[:-1, :, 2] = Z[1:, :] - Z[:-1, :]

    # Compute normals using cross product
    normals = torch.cross(Vx, Vy, dim=2)

    # Normalize the normals
    normals = torch.nn.functional.normalize(normals, dim=2)

    # Replace NaN or infinite values
    # normals = torch.nan_to_num(normals)

    return normals

def get_pcd(frame: Frame, bbox: Optional[o3d.geometry.OrientedBoundingBox] = None, mask_generator: Optional[SamAutomaticMaskGenerator] = None, segment_cache_path: Optional[Path] = None) -> LabelledPcd:
    
    assert frame.depth is not None, "Depth image is required to get point cloud"

    depth_img = frame.depth.cpu().numpy()
    #normals = normals_from_depth(frame.depth, frame.K)[frame.depth > 0].cpu().numpy()
    # this gives normals in camera space
    normals = normals_from_depth(frame.depth, frame.K)
    # convert to world space and numpy
    normals = normals.view(-1, 3) @ frame.X_WV_opencv[:3, :3].cuda().T
    normals = normals.reshape(frame.color.shape)[frame.depth > 0].cpu().numpy()

    color_image = frame.color.cpu().numpy()




    seg_path = None
    if segment_cache_path is not None:
        seg_path = segment_cache_path / frame.name
        seg_path.parent.mkdir(parents=True, exist_ok=True)
    

    if mask_generator is not None:
        group_ids = sam3d.get_sam(color_image, mask_generator)
        if seg_path is not None:
            img = Image.fromarray(num_to_natural(group_ids).astype(np.int16), mode='I;16')
            img.save(str(seg_path) + ".png")
    else:
        assert seg_path is not None, "No mask generator provided and no segment cache path provided"
        img = Image.open(str(seg_path) + ".png")
        group_ids = np.array(img, dtype=np.int16)

    # use open3d to create pointcloud from depth and col
    depth = o3d.geometry.Image(depth_img)
    color = o3d.geometry.Image((color_image * 255).astype(np.uint8))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1.0, depth_trunc=2.0, convert_rgb_to_intensity=False)
    
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(frame.w, frame.h, frame.fl_x, frame.fl_y, frame.cx, frame.cy)
    extrinsic = frame.X_VW_opencv.cpu().numpy()
    pcd_color = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic, project_valid_depth_only=True)

    group_ids = group_ids.astype(np.float32)
    group_ids = np.reshape(group_ids, (depth_img.shape[0], depth_img.shape[1], 1))
    groups = o3d.geometry.Image(group_ids)

    rgbd_groups = o3d.geometry.RGBDImage.create_from_color_and_depth(groups, depth, depth_scale=1.0, depth_trunc=2.0, convert_rgb_to_intensity=False)
    pcd_groups = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_groups, intrinsic, extrinsic, project_valid_depth_only=True)
    pcd_groups.normals = o3d.utility.Vector3dVector(normals.reshape(-1, 3))
    #pcd_groups.estimate_normals()

    if bbox is not None:
        point_indices = bbox.get_point_indices_within_bounding_box(pcd_groups.points)
        pcd_groups = pcd_groups.select_by_index(point_indices)
        pcd_groups.normals = o3d.utility.Vector3dVector(normals.reshape(-1, 3)[point_indices])
        pcd_color = pcd_color.select_by_index(point_indices)


    save_dict = dict(coord=np.array(pcd_color.points), color=np.array(pcd_color.colors), normals=np.array(pcd_groups.normals), group=np.array(pcd_groups.colors)[:,0].astype(np.int16))
    return save_dict # type: ignore


def seg_pcd(dataset, mask_generator: SamAutomaticMaskGenerator, voxelize: Voxelize, bbox: Optional[o3d.geometry.OrientedBoundingBox] = None, intermediate_outputs_path: Optional[Path] = None) -> LabelledPcd:
    
    pcd_list = []
    seg_path = None
    if intermediate_outputs_path is not None:
        seg_path = intermediate_outputs_path / "segments"

    for frame in dataset.frames:
        pcd_dict = get_pcd(frame, bbox=bbox, mask_generator=mask_generator, segment_cache_path=seg_path)
        if len(pcd_dict["coord"]) == 0:
            continue
        pcd_dict = voxelize(pcd_dict)
        pcd_list.append(pcd_dict)

    
    while len(pcd_list) != 1:
        print(f"merging {len(pcd_list)} point clouds", flush=True)
        new_pcd_list = []
        for indice in sam3d.pairwise_indices(len(pcd_list)):
            # print(indice)
            # voxel_size in cal_2_scenes is actually the distance threshold for merging point groups
            # which is dependent on the depth error of the sensor, it is multiplied by 1.5 before use
            depth_error = 0.03 # 3cm depth error
            pcd_frame = sam3d.cal_2_scenes(pcd_list, indice, voxel_size=max(VOXEL_SIZE, depth_error/1.5), voxelize=voxelize)
            if pcd_frame is not None:
                new_pcd_list.append(pcd_frame)
        pcd_list = new_pcd_list

    
    seg_dict = pcd_list[0]
    seg_dict["group"] = num_to_natural(sam3d.remove_small_group(seg_dict["group"], TH))

    return seg_dict



def get_object_meshes(pcd_dict: LabelledPcd, dataset: StaticDataset, scene: SceneSetup) -> List[o3d.geometry.TriangleMesh]:
    
    # first remove all points outside the workspace
    voxel_grid = get_workspace_voxels(scene)

    valid_mask = voxel_grid.check_if_included(o3d.utility.Vector3dVector(pcd_dict['coord']))
    pcd_dict = {k: v[valid_mask] for k, v in pcd_dict.items()} # type: ignore
    
    
    object_groups, _ = extract_object_and_table_groups(pcd_dict, dataset, scene)
    # we calculate the full mesh then delete items from it, this prevents possion method from creating large blobs for the unseen parts of each object
    full_mesh = construct_mesh(pcd_dict)
    meshes = []
    for group in object_groups:
        mesh = get_mesh_for_group(full_mesh, pcd_dict, group)
        meshes.append(mesh)
    return meshes



def groups_in_view(pcd: LabelledPcd, frame: Frame) -> List[int]:
    _, valid_mask = frame.project(torch.tensor(pcd['coord']).float())
    groups = np.unique(pcd['group'][valid_mask])
    return groups.tolist()


def construct_mesh(pcd_dict: LabelledPcd) -> o3d.geometry.TriangleMesh:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_dict['coord']))
    pcd.colors = o3d.utility.Vector3dVector(pcd_dict['color'])
    pcd.normals = o3d.utility.Vector3dVector(pcd_dict['normals'])
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    return mesh


def get_mesh_for_group(mesh: o3d.geometry.TriangleMesh, pcd_dict: LabelledPcd, group: int) -> o3d.geometry.TriangleMesh:
    coords = pcd_dict['coord'][pcd_dict['group'] == group]
    
    vertices = np.asarray(mesh.vertices)

    scene_coord = torch.tensor(vertices).cuda().contiguous().float()
    new_offset = torch.tensor(scene_coord.shape[0]).cuda()
    gen_coord = torch.tensor(coords).cuda().contiguous().float()
    offset = torch.tensor(gen_coord.shape[0]).cuda()
    _, dis = pointops.knn_query(1, gen_coord, offset, scene_coord, new_offset) # type: ignore
    
    #get indicies of all vertices
    vertices_indices = torch.arange(0, len(vertices)).cuda()
    indices = vertices_indices.cpu().numpy()
    
    mask_dis = dis.reshape(-1).cpu().numpy() <= VOXEL_SIZE     
    object_mesh = mesh.select_by_index(indices[mask_dis])

    #object_mesh = mesh.select_by_index(indices)
    #return clean_and_watertight_mesh(object_mesh)
    return object_mesh



def extract_valid_groups_by_viewability(segmented_cloud: LabelledPcd, dataset: StaticDataset) -> List[int]:
    visibilitiy = []
    for f in dataset.frames:
        groups = groups_in_view(segmented_cloud, f)
        visibilitiy.extend(groups)
    
    all_groups = np.unique(segmented_cloud["group"])
    thresh = len(dataset.frames) - 1
    valid_groups = [g for g in all_groups if visibilitiy.count(g) >= thresh and g >= 0]
    return valid_groups


def get_workspace_voxels(scene: SceneSetup) -> o3d.geometry.VoxelGrid:
    table_xyz = scene.ground_gaussians.xyz # ground_gaussians.xyz

    table_plane = scene.ground_plane
    table_normal = np.array([table_plane[0], table_plane[1], table_plane[2]])
    table_pcd_extruded = np.array(table_xyz).copy()

    DESIRED_HEIGHT = 1.0
    VOXEL_SIZE = 0.05
    iters = int(DESIRED_HEIGHT / VOXEL_SIZE)
    for i in range(iters):
        new_points = table_xyz + table_normal * VOXEL_SIZE * i
        table_pcd_extruded = np.append(table_pcd_extruded, new_points, axis=0)

    table_pcd_extruded = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(table_pcd_extruded))

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(table_pcd_extruded, VOXEL_SIZE*2)
    return voxel_grid

def extract_table_group(segmented_cloud: LabelledPcd, valid_groups: List[int]) -> int:
    # the table is the largest group so we use that.
    group_points = segmented_cloud["group"]
    
    # Find most frequent group from valid_groups
    unique, counts = np.unique(group_points, return_counts=True)
    freq_dict = dict(zip(unique, counts))
    table_group = max((freq_dict[g], g) for g in valid_groups)[1]
    return table_group

def extract_object_and_table_groups(segmented_cloud: LabelledPcd, dataset: StaticDataset, scene: SceneSetup) -> Tuple[List[int], int]:
    valid_groups = extract_valid_groups_by_viewability(segmented_cloud, dataset)
    logger.info(f"Valid groups: {valid_groups}")
    logger.info(f"removed : {len(np.unique(segmented_cloud['group'])) - len(valid_groups)} groups")
    
    
    table_group = extract_table_group(segmented_cloud, valid_groups)
    logger.info(f"Table group: {table_group}")

    return [g for g in valid_groups if g != table_group], table_group


def get_bounding_box_corners_from_gaussians(gaussians: GaussiansDef, density: float = 0.5) -> np.ndarray:
    xyz = gaussians.xyz  # Shape: [N, 3]
    scales = gaussians.scaling  # Shape: [N, 3]
    quaternions = gaussians.rotations  # Shape: [N, 4]

    # Calculate the extent of the bounding box based on the density
    extent = scales * np.sqrt(-2 * np.log(1 - density))  # Shape: [N, 3]

    # Create the 8 corners of the bounding box in local space
    local_corners = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [-1, 1, -1],
        [1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1]
    ])  # Shape: [8, 3]

    # Scale the local corners by the extent
    local_corners = local_corners * extent[:, np.newaxis, :]  # Shape: [N, 8, 3]

    # Convert quaternions to rotation matrices
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    rotation_matrices = np.stack([
        1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w,
        2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w,
        2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2
    ], axis=-1).reshape(-1, 3, 3)  # Shape: [N, 3, 3]

    # Apply rotation to the corners
    rotated_corners = np.einsum('nij,nkj->nki', rotation_matrices, local_corners)  # Shape: [N, 8, 3]

    # Translate the corners to the position of the Gaussians
    world_corners = rotated_corners + xyz[:, np.newaxis, :]  # Shape: [N, 8, 3]

    return world_corners


def get_scene_bounding_box(scene: SceneSetup) -> o3d.geometry.OrientedBoundingBox:
    # get bounding box from gaussian points
    # combine robot and ground gaussians
    all_gaussians = GaussiansDef(   
        xyz=np.concatenate([scene.robot_gaussians.xyz, scene.ground_gaussians.xyz]),
        scaling=np.concatenate([scene.robot_gaussians.scaling, scene.ground_gaussians.scaling]),
        rotations=np.concatenate([scene.robot_gaussians.rotations, scene.ground_gaussians.rotations]),
        opacity=np.concatenate([scene.robot_gaussians.opacity, scene.ground_gaussians.opacity]),
        colors=np.concatenate([scene.robot_gaussians.colors, scene.ground_gaussians.colors])
    )

    gaussian_corners = get_bounding_box_corners_from_gaussians(all_gaussians)
    
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gaussian_corners.reshape(-1, 3)))
    obb = o3d_pcd.get_oriented_bounding_box()
    return obb



def initialize_scene(dataset: StaticDataset, scene: SceneSetup, intermediate_outputs_path: Optional[Path] = None) -> ObjectsSpatialDef:
    mask_generator = SamAutomaticMaskGenerator( build_sam(checkpoint=sam_checkpoint).to(device="cuda"))
    voxelize = Voxelize(voxel_size=VOXEL_SIZE, mode="train", keys=("coord", "color", "group", "normals"))


    # extract bounding box from scene
    obb = get_scene_bounding_box(scene)

    segmented_cloud = seg_pcd(dataset, mask_generator, voxelize, bbox=obb, intermediate_outputs_path=intermediate_outputs_path)
    
    logger.info(f"Segmented cloud has {np.unique(segmented_cloud['group'])} unique groups - {np.unique(segmented_cloud['group'])}")

    # TODO: remove points on wrong side of table plane
    object_meshes = get_object_meshes(segmented_cloud, dataset, scene)

    clean_meshes = []
    valid_object_meshes = []

    for i, mesh in enumerate(object_meshes):
        clean_mesh = clean_and_watertight_mesh(mesh)
        if clean_mesh.has_triangles():
            valid_object_meshes.append(mesh)
            clean_meshes.append(clean_mesh)
        else:
            logger.warning(f"Mesh {i} is not clean and watertight, ignoring")

    # create particles from meshes
    object_particles = [fill_mesh_with_particles(mesh, PARTICLE_RADIUS) for mesh in clean_meshes]

    # create gaussians from meshes
    object_gaussians = [mesh_to_gaussians(mesh) for mesh in valid_object_meshes]

    objects = []
    for i, (particles, gaussians) in enumerate(zip(object_particles, object_gaussians)):
        objects.append(ObjectSpatialDef(
            object_id=i+1,
            particles=particles,
            gaussians=gaussians
        ))

    return ObjectsSpatialDef(
        objects=objects
    )



