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

import torch
import cv2
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def Depth_Normalize(depth_tensor):
    """
    Converts a single-channel depth image tensor to a color mapped version for visualization.
    """
    # Normalize the depth image to 0-1 for better color mapping
    depth_min = depth_tensor.min()
    depth_max = depth_tensor.max()
    depth_normalized = (depth_tensor - depth_min) / (depth_max - depth_min + 1e-5)

    # Apply colormap
    # Convert tensor to numpy array and ensure it's in CPU
    # depth_numpy = depth_normalized.cpu().numpy()
    
    return depth_normalized



def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_depth")
    render_depth_alpha_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_depth_alpha")
    render_alpha_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_alpha")
    
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gt_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depth")
    
    
    makedirs(render_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_depth_alpha_path, exist_ok=True)
    makedirs(render_alpha_path, exist_ok=True)
    
    makedirs(gts_path, exist_ok=True)
    makedirs(gt_depth_path, exist_ok=True)
    

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        render_depth = render_pkg["depth"]
        render_alpha = render_pkg["alpha"]
        
        render_depth_normalized = Depth_Normalize(render_depth)
        render_depth_alpha_normalized = Depth_Normalize(render_depth / render_alpha)
        gt = view.original_image[0:3, :, :]
        gt_depth = Depth_Normalize(view.original_depth)
        print(f"gt_depth: {gt_depth.shape} and render_depth: {render_depth.shape}")
        #print(f"{render_depth_normalized.max()} and {gt_depth.max()}\n")
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt_depth, os.path.join(gt_depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(render_depth_normalized, os.path.join(render_depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(render_depth_alpha_normalized, os.path.join(render_depth_alpha_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(render_alpha, os.path.join(render_alpha_path, '{0:05d}'.format(idx) + ".png"))
        
        #render_depth_normalized = render_depth_normalized * 255.0
        #render_depth_normalized = render_depth_normalized.cpu().numpy().astype(np.uint8)
        #cv2.imwrite(os.path.join(render_depth_path, '{0:05d}'.format(idx) + ".png"), render_depth_normalized * 255.0)
        

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)