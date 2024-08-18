#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
from scene.gaussian_model import BasicPointCloud
from .dataset_readers import getNerfppNorm, fetchPly, storePly

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth:np.array
    image_path: str
    image_name: str
    width: int
    height: int
    K: np.array
    cx: float
    cy: float
    fx: float
    fy: float


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, depths_folder, camera_num=1):
    cam_infos = []
    frame_len = int(len(cam_extrinsics)/camera_num)
    frame_start = 90
    frame_end = 197
    selected_frames = []
    for i in range(camera_num):
        for j in range(frame_start, frame_end):
            selected_frames.append(frame_len*i+j)
    for idx, key in enumerate(cam_extrinsics):
        if idx not in selected_frames:
            continue
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec) + np.array([2., 0, 0])
        
        K = None
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            K = intr.params
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        fx = float(intr.params[0])
        fy = float(intr.params[1])
        cx = float(intr.params[2])
        cy = float(intr.params[3])
        image_path = os.path.join(images_folder, extr.name)
        depth_path = os.path.join(depths_folder, extr.name.replace(".jpg", ".npy"))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        #depth = np.load(depth_path)
        depth = Image.fromarray(depth)
        
        
        #depth_path = os.path.join(path, 'lidar_depth', f'{image_name}.npy')
        
        depth = np.load(depth_path, allow_pickle=True)
        if isinstance(depth, np.ndarray):
            depth = dict(depth.item())
            mask = depth['mask']
            value = depth['value']
            depth = np.zeros_like(mask).astype(np.float32)
            depth[mask] = value
        
        
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth,
                              image_path=image_path, image_name=image_name, width=width, height=height, K=K,
                              cx=cx, cy=cy, fx=fx, fy=fy)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readWaymoSceneInfo(path, images, eval, filter_camera_id=[1],llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "colmap/sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "colmap/sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file, filter_camera_id=filter_camera_id)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "colmap/sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "colmap/sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), 
                                           depths_folder=os.path.join(path, "depths"), camera_num=len(filter_camera_id))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "colmap/sparse/0/points3D.ply")
    bin_path = os.path.join(path, "colmap/sparse/0/points3D.bin")
    txt_path = os.path.join(path, "colmap/sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
