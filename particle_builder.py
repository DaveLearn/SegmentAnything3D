from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
from psdstaticdataset import StaticDataset
from psdframe import Frame
from initializerdefs import SceneSetup, ObjectsSpatialDef, ParticlesSpatialDef,ObjectSpatialDef
from PIL import Image
from mesh_to_gaussians import mesh_to_gaussians
import sam3d
from download_sam import get_sam_checkpoint
from segment_anything import build_sam, SamAutomaticMaskGenerator
from util import Voxelize, num_to_natural
import numpy as np
import os
import cv2
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

def get_pcd(frame: Frame, mask_generator: Optional[SamAutomaticMaskGenerator] = None, segment_cache_path: Optional[Path] = None) -> LabelledPcd:
    
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

    save_dict = dict(coord=np.array(pcd_color.points), color=np.array(pcd_color.colors), normals=np.array(pcd_groups.normals), group=np.array(pcd_groups.colors)[:,0].astype(np.int16))
    return save_dict


def seg_pcd(dataset, mask_generator: SamAutomaticMaskGenerator, voxelize: Voxelize, intermediate_outputs_path: Optional[Path] = None) -> LabelledPcd:
    
    pcd_list = []
    seg_path = None
    if intermediate_outputs_path is not None:
        seg_path = intermediate_outputs_path / "segments"

    for frame in dataset.frames:
        pcd_dict = get_pcd(frame, mask_generator, segment_cache_path=seg_path)
        if len(pcd_dict["coord"]) == 0:
            continue
        pcd_dict = voxelize(pcd_dict)
        pcd_list.append(pcd_dict)

    
    while len(pcd_list) != 1:
        print(f"merging {len(pcd_list)} point clouds", flush=True)
        new_pcd_list = []
        for indice in sam3d.pairwise_indices(len(pcd_list)):
            # print(indice)
            pcd_frame = sam3d.cal_2_scenes(pcd_list, indice, voxel_size=VOXEL_SIZE, voxelize=voxelize)
            if pcd_frame is not None:
                new_pcd_list.append(pcd_frame)
        pcd_list = new_pcd_list

    
    seg_dict = pcd_list[0]
    seg_dict["group"] = num_to_natural(sam3d.remove_small_group(seg_dict["group"], TH))

    return seg_dict



def get_object_meshes(pcd_dict: LabelledPcd, dataset: StaticDataset) -> List[o3d.geometry.TriangleMesh]:
    object_groups, _ = extract_object_and_table_groups(pcd_dict, dataset)
    # we calculate the full mesh then delete items from it, this prevents possion method from creating large blobs for the unseen parts of each object
    full_mesh = construct_mesh(pcd_dict)
    meshes = []
    for group in object_groups:
        mesh = get_mesh_for_group(full_mesh, pcd_dict, group)
        meshes.append(mesh)
    return meshes



def groups_in_view(pcd: LabelledPcd, frame: Frame) -> List[int]:
    _, valid_mask = frame.project(pcd['coord'])
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
    _, dis = pointops.knn_query(1, gen_coord, offset, scene_coord, new_offset)
    
    #get indicies of all vertices
    vertices_indices = torch.arange(0, len(vertices)).cuda()
    indices = vertices_indices.cpu().numpy()
    
    mask_dis = dis.reshape(-1).cpu().numpy() <= VOXEL_SIZE     
    object_mesh = mesh.select_by_index(indices[mask_dis])

    #object_mesh = mesh.select_by_index(indices)
    #return clean_and_watertight_mesh(object_mesh)
    return object_mesh






def extract_object_and_table_groups(segmented_cloud: LabelledPcd, dataset: StaticDataset) -> Tuple[List[int], int]:
    visibilitiy = []
    for f in dataset.frames:
        groups = groups_in_view(segmented_cloud, f)
        visibilitiy.extend(groups)

    all_groups = np.unique(segmented_cloud["group"])
    thresh = len(dataset.frames) - 1
    valid_groups = [g for g in all_groups if visibilitiy.count(g) >= thresh and g >= 0]
    logger.info(f"Valid groups: {valid_groups}")
    logger.info(f"removed : {len(all_groups) - len(valid_groups)} groups")
    # remove the table which will be the largest group
    group_points = segmented_cloud["group"]
    
    # Find most frequent group from valid_groups
    unique, counts = np.unique(group_points, return_counts=True)
    freq_dict = dict(zip(unique, counts))
    table_group = max((freq_dict[g], g) for g in valid_groups)[1]

    return [g for g in valid_groups if g != table_group], table_group


def initialize_scene(dataset: StaticDataset, scene: Optional[SceneSetup] = None, intermediate_outputs_path: Optional[Path] = None) -> ObjectsSpatialDef:
    mask_generator = SamAutomaticMaskGenerator( build_sam(checkpoint=sam_checkpoint).to(device="cuda"))
    voxelize = Voxelize(voxel_size=VOXEL_SIZE, mode="train", keys=("coord", "color", "group", "normals"))
    segmented_cloud = seg_pcd(dataset, mask_generator, voxelize, intermediate_outputs_path)
    
    logger.info(f"Segmented cloud has {np.unique(segmented_cloud['group'])} unique groups - {np.unique(segmented_cloud['group'])}")

    # TODO: remove points on wrong side of table plane
    object_meshes = get_object_meshes(segmented_cloud, dataset)

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



