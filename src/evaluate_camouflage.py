import argparse
import logging
import os
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from pathlib import Path

import neural_renderer

from camouflage_eval import evaluate_camouflage_dataset
from rauca_utils.datasets_RAUCA import create_dataloader
from rauca_utils.general_RAUCA import check_dataset, colorstr, set_logging, increment_path
from models.experimental import attempt_load
from utils.torch_utils import select_device


def cal_texture(texture_param, texture_origin, texture_mask):
    textures = 0.5 * (torch.tanh(texture_param) + 1)
    return texture_origin * (1 - texture_mask) + texture_mask * textures


def build_texture_mask(faces, texture_size, faces_file, device, logger=None):
    """
    Build mask for textures.
    - If faces_file lines are single integers (like training), mark the whole face.
    - If lines are comma-separated axis-range (face_id, axis, start, end), only mark that slice.
    """
    texture_mask = np.zeros((faces.shape[0], texture_size, texture_size, texture_size, 3), dtype=np.int8)
    with open(faces_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(',')
            if len(parts) == 1 and parts[0]:
                # training-style: only face id
                face_id = int(parts[0])
                texture_mask[face_id - 1, :, :, :, :] = 1
            elif len(parts) >= 4:
                face_id = int(parts[0])
                axis = int(parts[1])
                start = int(parts[2])
                end = int(parts[3])
                texture_mask[face_id - 1, axis, start:end, :, :] = 1
            else:
                continue
    texture_mask = torch.from_numpy(texture_mask).to(device).unsqueeze(0)
    return texture_mask


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/carla.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default='weights/yolov7-e6e.pt', help='YOLOv7 weights')
    parser.add_argument('--texture-npy', type=str, required=True, help='trained texture_param npy file')
    parser.add_argument('--obj-file', type=str, default='car_assets/audi_et_te.obj', help='3D car model obj path')
    parser.add_argument('--vertex-offset-x', type=float, default=0.0, help='x-axis offset applied to vertices')
    parser.add_argument('--vertex-offset-y', type=float, default=0.0, help='y-axis offset applied to vertices')
    parser.add_argument('--vertex-offset-z', type=float, default=0.33, help='z-axis offset applied to vertices')
    parser.add_argument('--faces', type=str, default='car_assets/exterior_face.txt', help='face id file')
    parser.add_argument('--datapath', type=str, default='data/dataset', help='dataset root path')
    parser.add_argument('--mask-dir', type=str, default='', help='mask directory (defaults to <datapath>/masks/)')
    parser.add_argument('--texturesize', type=int, default=6, help='texture size')
    parser.add_argument('--img-size', type=int, default=640, help='inference image size')
    parser.add_argument('--batch-size', type=int, default=1, help='dataloader batch size')
    parser.add_argument('--device', default='', help='cuda device or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--camou-scale', type=float, default=1.7, help='camouflage renderer scale')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--phase', type=str, default='test', choices=['training', 'test'], help='dataset phase')
    parser.add_argument('--save-dir', type=str, default='runs/camo_eval', help='output directory')
    parser.add_argument('--single-cls', action='store_true', help='treat as single class')
    return parser.parse_args()


def main(opt):
    set_logging()
    logger = logging.getLogger(__name__)
    device = select_device(opt.device)

    with open(opt.data, errors='ignore') as f:
        data_dict = yaml.safe_load(f)
    check_dataset(data_dict)
    names = data_dict.get('names', [str(i) for i in range(data_dict.get('nc', 1))])

    dataset_path = data_dict['val'] or data_dict['train']
    mask_dir = opt.mask_dir or os.path.join(opt.datapath, 'masks/')

    vertices, faces, texture_origin = neural_renderer.load_obj(filename_obj=opt.obj_file,
                                                               texture_size=opt.texturesize,
                                                               load_texture=True)
    if opt.vertex_offset_x or opt.vertex_offset_y or opt.vertex_offset_z:
        offset = torch.tensor([opt.vertex_offset_x,
                               opt.vertex_offset_y,
                               opt.vertex_offset_z],
                              dtype=vertices.dtype,
                              device=vertices.device)
        vertices = vertices + offset
    faces = faces.to(device)
    vertices = vertices.to(device)
    texture_origin = texture_origin.to(device)

    texture_param_np = np.load(opt.texture_npy, allow_pickle=True)
    texture_param = torch.from_numpy(texture_param_np).to(device)
    texture_mask = build_texture_mask(faces, opt.texturesize, opt.faces, device, logger=logger)
    adv_textures = cal_texture(texture_param, texture_origin, texture_mask)

    model = attempt_load(opt.weights, map_location=device)
    gs = max(int(model.stride.max()), 32)

    runner_opt = SimpleNamespace(single_cls=opt.single_cls,
                                 cache_images=False,
                                 quad=False,
                                 mask_dir=mask_dir,
                                 camou_scale=opt.camou_scale,
                                 world_size=1,
                                 workers=opt.workers)

    _, eval_dataset = create_dataloader(dataset_path,
                                        opt.img_size,
                                        opt.batch_size,
                                        gs,
                                        faces,
                                        opt.texturesize,
                                        vertices,
                                        runner_opt,
                                        hyp=None,
                                        augment=False,
                                        cache=False,
                                        pad=0.0,
                                        rect=False,
                                        rank=-1,
                                        world_size=1,
                                        workers=opt.workers,
                                        prefix=colorstr('eval: '),
                                        mask_dir=mask_dir,
                                        ret_mask=True,
                                        camou_scale=opt.camou_scale,
                                        phase=opt.phase)

    save_dir = increment_path(Path(opt.save_dir) / 'exp', exist_ok=False)
    clean_tex = texture_origin.detach().clone()
    evaluate_camouflage_dataset(model,
                                eval_dataset,
                                device,
                                names,
                                save_dir,
                                clean_tex,
                                adv_textures,
                                conf_thres=opt.conf_thres,
                                iou_thres=opt.iou_thres,
                                logger=logger,
                                clean_label='texture_origin',
                                adv_label=str(Path(opt.texture_npy).resolve()))
    logger.info(f"Evaluation complete. Outputs saved to {save_dir.resolve()}")


if __name__ == '__main__':
    opts = parse_opt()
    main(opts)
