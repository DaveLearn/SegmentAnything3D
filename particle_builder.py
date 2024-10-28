from pathlib import Path
from typing import List, Optional, TypedDict
from psdstaticdataset import StaticDataset
from psdframe import Frame
from initializerdefs import SceneSetup, ObjectsSpatialDef
from PIL import Image
import sam3d
from download_sam import get_sam_checkpoint
from segment_anything import build_sam, SamAutomaticMaskGenerator
from util import Voxelize, num_to_natural
import numpy as np
import os
import cv2
import open3d as o3d

import logging
logger = logging.getLogger("sam3d-builder")

logger.info("Downloading/Locating SAM checkpoint...")
sam_checkpoint = get_sam_checkpoint()
logger.info(f"Sam checkpoint in located at {sam_checkpoint}")

VOXEL_SIZE = 0.01
TH = 50

class LabelledPcd(TypedDict):
    coord: np.ndarray
    color: np.ndarray
    group: np.ndarray


def get_pcd(frame: Frame, mask_generator: Optional[SamAutomaticMaskGenerator] = None, segment_cache_path: Optional[Path] = None) -> LabelledPcd:
    
    depth_img = frame.depth.cpu().numpy()
    mask = (depth_img != 0)
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

   # color_image = np.reshape(color_image[mask], color_image.shape)
    #group_ids = group_ids[mask]


    # use open3d to create pointcloud from depth and col
    depth = o3d.geometry.Image(depth_img)
    color = o3d.geometry.Image((color_image * 255).astype(np.uint8))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False)
    
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(frame.w, frame.h, frame.fl_x, frame.fl_y, frame.cx, frame.cy)
    extrinsic = frame.X_VW_opencv.cpu().numpy()
    pcd_color = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic, project_valid_depth_only=True)

    # convert group_ids to float64
    group_ids = group_ids.astype(np.float32)
    # expand group_ids to 3 channels
    #group_ids = np.expand_dims(group_ids, axis=1)           
    #group_ids = np.concatenate([group_ids, group_ids, group_ids], axis=1)
    # reshape to 3 channel image
    group_ids = np.reshape(group_ids, (depth_img.shape[0], depth_img.shape[1], 1))
    groups = o3d.geometry.Image(group_ids)

    rgbd_groups = o3d.geometry.RGBDImage.create_from_color_and_depth(groups, depth, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False)
    pcd_groups = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_groups, intrinsic, extrinsic, project_valid_depth_only=True)

    save_dict = dict(coord=np.array(pcd_color.points), color=np.array(pcd_color.colors), group=np.array(pcd_groups.colors)[:,0].astype(np.int16))
    return save_dict
    



    colors = np.zeros_like(color_image)
    colors[:,0] = color_image[:,2]
    colors[:,1] = color_image[:,1]
    colors[:,2] = color_image[:,0]


    pose = frame.X_VW_opencv.cpu().numpy() #np.loadtxt(pose)
    depth_intrinsic = np.array([[frame.fl_x, 0, frame.cx], [0, frame.fl_y, frame.cy], [0, 0, 1]])
    
    x,y = np.meshgrid(np.linspace(0,depth_img.shape[1]-1,depth_img.shape[1]), np.linspace(0,depth_img.shape[0]-1,depth_img.shape[0]))
    uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
    uv_depth[:,:,0] = x
    uv_depth[:,:,1] = y
    uv_depth[:,:,2] = depth_img
    uv_depth = np.reshape(uv_depth, [-1,3])
    uv_depth = uv_depth[np.where(uv_depth[:,2]!=0),:].squeeze()
    
   
    fx = depth_intrinsic[0,0]
    fy = depth_intrinsic[1,1]
    cx = depth_intrinsic[0,2]
    cy = depth_intrinsic[1,2]
    bx = 0 # depth_intrinsic[0,3]
    by = 0 # depth_intrinsic[1,3]
    n = uv_depth.shape[0]
    points = np.ones((n,4))
    X = (uv_depth[:,0]-cx)*uv_depth[:,2]/fx + bx
    Y = (uv_depth[:,1]-cy)*uv_depth[:,2]/fy + by
    points[:,0] = X
    points[:,1] = Y
    points[:,2] = uv_depth[:,2]

    print('4')
    #points_world = np.dot(points, np.transpose(pose))
    points_world = np.dot(points, np.transpose(pose))
    group_ids = num_to_natural(group_ids)
    save_dict = dict(coord=points_world[:,:3], color=colors, group=group_ids)
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




def groups_in_view(pcd: LabelledPcd, frame: Frame) -> List[int]:
    # using the frames pose, find which groups of the PCD are visible from the frame
    # Convert frame pose to camera matrix
    camera_matrix = frame.pose.to_matrix()
    
    # Extract camera position and orientation
    camera_position = camera_matrix[:3, 3]
    camera_forward = -camera_matrix[:3, 2]  # Negative z-axis is the camera direction
    camera_right = camera_matrix[:3, 0]
    camera_up = camera_matrix[:3, 1]
    
    # Calculate vectors from camera to all points
    vectors_to_points = pcd['coord'] - camera_position
    
    # Normalize vectors
    distances = np.linalg.norm(vectors_to_points, axis=1)
    normalized_vectors = vectors_to_points / distances[:, np.newaxis]
    
    # Calculate angles
    cos_vertical = np.dot(normalized_vectors, camera_up)
    cos_horizontal = np.dot(normalized_vectors, camera_right)
    
    # Convert to angles
    vertical_angles = np.arccos(cos_vertical)
    horizontal_angles = np.arccos(cos_horizontal)
    
    # Calculate field of view based on focal lengths
    # Assuming sensor width and height in mm
    sensor_width = 36  # for a full-frame sensor, adjust if different
    sensor_height = 24  # for a full-frame sensor, adjust if different
    fov_horizontal = 2 * np.arctan(sensor_width / (2 * frame.camera.fl_x))
    fov_vertical = 2 * np.arctan(sensor_height / (2 * frame.camera.fl_y))
    
    # Define visibility criteria
    max_distance = 10.0  # Maximum visible distance
    
    # Determine visible points using angles and distance
    visible_mask = (np.dot(normalized_vectors, camera_forward) > 0) & \
                   (distances < max_distance) & \
                   (np.abs(horizontal_angles - np.pi/2) < fov_horizontal/2) & \
                   (np.abs(vertical_angles - np.pi/2) < fov_vertical/2)
    
    # Get unique groups of visible points
    visible_groups = np.unique(pcd['group'][visible_mask])
    
    return visible_groups.tolist()

def initialize_scene(dataset: StaticDataset, scene: SceneSetup, intermediate_outputs_path: Optional[Path] = None) -> ObjectsSpatialDef:
    mask_generator = SamAutomaticMaskGenerator( build_sam(checkpoint=sam_checkpoint).to(device="cuda"))
    voxelize = Voxelize(voxel_size=VOXEL_SIZE, mode="train", keys=("coord", "color", "group"))
    segmented_cloud = seg_pcd(dataset, mask_generator, voxelize, intermediate_outputs_path)


