import os
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from os import makedirs
import torchvision
from arguments import get_combined_args

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def Depth_Normalize(depth_tensor):
    depth_min = depth_tensor.min()
    depth_max = depth_tensor.max()
    depth_normalized = (depth_tensor - depth_min) / (depth_max - depth_min + 1e-5)

    return depth_normalized



def render_sfm(render_sfm_path, views, gaussians, pipeline, background):
    sfm_depth_path = os.path.join(render_sfm_path,"sfm_depth")
    sfm_render_path = os.path.join(render_sfm_path,"sfm_render")
    makedirs(sfm_depth_path, exist_ok=True)
    makedirs(sfm_render_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering SfM")):
        render_pkg = render(view, gaussians, pipeline, background)
        sfm_depth = render_pkg["depth"]
        sfm_alpha = render_pkg["alpha"]
        alpha_mask = (sfm_alpha > 0.8)
        
        result_depth = torch.zeros_like(sfm_depth)
        
        result_depth[alpha_mask] = sfm_depth[alpha_mask] / sfm_alpha[alpha_mask]
        
        sfm_render = render_pkg["render"]
        print(f"result_depth{result_depth.shape}")
        print(f"max {result_depth.max()}")
        #print(result_depth)
        save_path = sfm_depth_path
        if idx <= 60:
            save_path = os.path.join(sfm_depth_path, '_DSC{0:04d}'.format(idx + 8679) + ".npy")
            torchvision.utils.save_image(Depth_Normalize(result_depth), os.path.join(sfm_depth_path, '_DSC{0:04d}'.format(idx + 8679) + ".png"))
        else:
            save_path = os.path.join(sfm_depth_path, '_DSC{0:04d}'.format(idx + 8680) + ".npy")
            torchvision.utils.save_image(Depth_Normalize(result_depth), os.path.join(sfm_depth_path, '_DSC{0:04d}'.format(idx + 8680) + ".png"))
        sfm_depth_numpy = result_depth.cpu().detach().numpy()
        np.save(save_path, sfm_depth_numpy)
        #torchvision.utils.save_image(Depth_Normalize(result_depth), os.path.join(sfm_depth_path, '_DSC{0:04d}'.format(idx + 8679) + ".png"))
        #torchvision.utils.save_image(sfm_render, save_path)
        
        #torchvision.utils.save_image(result_depth, os.path.join(sfm_depth_path, '_DSC{0:04d}'.format(idx + 8680) + ".npy"))
        #torchvision.utils.save_image(sfm_render, os.path.join(sfm_render_path, '_DSC{0:04d}'.format(idx + 8680) + ".png"))
def make_sfm_pics(dataset, opt, pipe):
    gaussians = GaussianModel(dataset.sh_degree)
    print(f"Shape of gaussians.get_xyz#1: {gaussians.get_xyz.shape}")
    scene = Scene(dataset, gaussians, sparse=True)
    gaussians.training_setup(opt)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    print(f"Shape of gaussians.get_xyz#2: {gaussians.get_xyz.shape}")
    render_sfm_path = "/nas/home/pengziyue/datasets/mipnerf360/bicycle"
    render_sfm(render_sfm_path, scene.getTrainCameras(), gaussians, pipe, background)
    
if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    print("Optimizing " + args.model_path)
    # Start GUI server, configure and run training
    #network_gui.init(args.ip, args.port)
    #torch.autograd.set_detect_anomaly(args.detect_anomaly)
    print("Init done!!!")
    make_sfm_pics(lp.extract(args), op.extract(args), pp.extract(args))

    # All done
    print("\nTraining complete.")
    