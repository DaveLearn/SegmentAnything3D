from pathlib import Path
from typing import List, Optional, TypedDict
from psdstaticdataset import StaticDataset
from psdframe import Frame
from initializerdefs import SceneSetup, ObjectsSpatialDef
import sam3d
from download_sam import get_sam_checkpoint
from segment_anything import build_sam, SamAutomaticMaskGenerator
from util import Voxelize
import numpy as np

import logging
logger = logging.getLogger("sam3d-builder")

logger.info("Downloading/Locating SAM checkpoint...")
sam_checkpoint = get_sam_checkpoint()
logger.info(f"Sam checkpoint in located at {sam_checkpoint}")

VOXEL_SIZE = 0.05
TH = 50

class LabelledPcd(TypedDict):
    coord: np.ndarray
    color: np.ndarray
    group: np.ndarray


def seg_pcd(dataset, mask_generator, voxelize, intermediate_outputs_path) -> LabelledPcd:
    pass

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
    mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=sam_checkpoint).to(device="cuda"))
    voxelize = Voxelize(voxel_size=VOXEL_SIZE, mode="train", keys=("coord", "color", "group"))
    segmented_cloud = seg_pcd(dataset, mask_generator, voxelize, intermediate_outputs_path)


