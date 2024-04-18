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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import generate_ellipse_path, pose_spherical
from utils.graphics_utils import getWorld2View2
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
import time

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    renderings = []
    t_list = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]

        renderings.append(to8b(gt.cpu().numpy()))

    #renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    #imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=24, quality=8)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()
        t_start = time.time()

        rendering = render(view, gaussians, pipeline, background)["render"]

        torch.cuda.synchronize()
        t_end = time.time()
        t_list.append(t_end - t_start)

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')


def interpolate_all(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 520
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
                               0)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        # torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)


def interpolate_view_original(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 480
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    R = []
    T = []
    for idx, view in enumerate(views):
        if idx in [26, 29, 31, 33, 36, 38, 40, 43, 49, 51, 53]:
            R.append(view.R)
            T.append(view.T)

    view = views[0]
    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        query_idx = i / frame * len(R)
        begin_idx = int(np.floor(query_idx))
        end_idx = int(np.ceil(query_idx))
        if end_idx == len(R):
            break
        R_begin = R[begin_idx]
        R_end = R[end_idx]
        t_begin = T[begin_idx]
        t_end = T[end_idx]

        ratio = query_idx - begin_idx

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)


def pose_distance(R1, T1, R2, T2):
    rotation_diff = np.linalg.norm(R2 - R1)
    translation_diff = np.linalg.norm(T2 - T1)
    return rotation_diff + translation_diff

def interpolate_view_original2(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 480
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    R = []
    T = []
    for idx, view in enumerate(views):
        if idx in [26, 29, 31, 33, 36, 38, 40, 43, 49, 51, 53]:
            R.append(view.R)
            T.append(view.T)

    view = views[0]
    renderings = []
    distances = [pose_distance(R[i], T[i], R[i+1], T[i+1]) for i in range(len(R)-1)]
    total_distance = sum(distances)
    frames_per_segment = [int(round(frame * d / total_distance)) for d in distances]

    current_frame = 0

    for segment in tqdm(range(len(frames_per_segment)), desc="Rendering progress"):
        for i in range(frames_per_segment[segment]):
            ratio = i / frames_per_segment[segment]
            R_cur = (1 - ratio) * R[segment] + ratio * R[segment + 1]
            T_cur = (1 - ratio) * T[segment] + ratio * T[segment + 1]

            view.reset_extrinsic(R_cur, T_cur)

            results = render(view, gaussians, pipeline, background)
            rendering = results["render"]
            renderings.append(to8b(rendering.cpu().numpy()))
            depth = results["depth"]
            depth = depth / (depth.max() + 1e-5)

            current_frame += 1
            if current_frame >= frame:
                break

        if current_frame >= frame:
            break

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)


def render_video(model_path, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, 'video', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    view = views[0]
    renderings = []
    for idx, pose in enumerate(tqdm(generate_ellipse_path(views,n_frames=600), desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        rendering = render(view, gaussians, pipeline, background)["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        #torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        #render_video(dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
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
