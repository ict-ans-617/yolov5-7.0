# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path
import time
from train import RANK
from train import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from thop import profile, clever_format
from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (ModelEMA,fuse_conv_and_bn, de_parallel, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

from val import get_val_result

import logging
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

from nni.compression.pruning import (
    L1NormPruner,
    L2NormPruner,
    FPGMPruner,
    AGPPruner
)
from nni.compression.speedup import ModelSpeedup
from nni.compression.speedup import replacer
replacer._logger.setLevel(logging.WARNING)
import torch
from torchsummary import summary
from models.yolo import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s_voc.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')

    #prune
    parser.add_argument('--pruner', default='fpgm', type=str, help='pruner: agp|taylor|fpgm')
    parser.add_argument('--pre_weights', type=str, default="./runs/train/exp89/weights/best.pt", help='pretrain weights for prune')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='directory to output')

    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)
    torch.cuda.empty_cache()

    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    # model = Model(opt.cfg).to(device) #从cfg创建model，并随机初始化权重
    ckpt = torch.load(opt.pre_weights,map_location='cpu')  # 加载预训练权重
    model = ckpt['model'].to(device)  # 获取模型对象并将其移动到指定设备
    model.float()  # 将模型参数转换为全精度浮点数
    original_model_size = os.path.getsize(opt.pre_weights) / (1024 * 1024)  # 将字节转换为MB
    print(f"剪枝之前模型 {opt.pre_weights} 的存储占用大小: {original_model_size:.2f} MB")

    val_outputs = get_val_result(weights=opt.pre_weights, device=opt.device)
    print(f"{val_outputs = }")
    # Options
    if opt.line_profile:  # profile layer by layer
        print("profile layer by layer")
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        print("profile forward-backward")
        results = profile(input=im, ops=[model], n=3)


    elif opt.test:  # test all models
        print("test all models")
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        print("report fused model summary")
        model.fuse()

    print('model paramater number: ', sum([param.numel() for param in model.parameters()]))
    

    #剪枝配置
    config_list = [{
                    # 'op_types': ['Conv2d'],
                    'op_types': ['Conv2d','BatchNorm2d'],
                    'sparse_ratio': 0.15, 
                    # 'op_names': [],
                    'exclude_op_names': ['model.24.m.0','model.24.m.1','model.24.m.2']
                    }]
    dummy_input = torch.rand(1, 3, 640, 640).to(device)

    if opt.pruner == 'l1':
        pruner = L1NormPruner(model, config_list)
        print('Prune type：L1NormPruner')
    elif opt.pruner == 'l2':
        pruner = L2NormPruner(model, config_list)
        print('Prune type：L2NormPruner')
    # elif opt.pruner == 'agp':
    #     config_list = [{'sparsity': args.sparsity, 'op_types': ['Conv2d']}]
    #     pruner = AGPPruner(
    #         model,
    #         config_list,
    #         optimizer = opt.optimizer,
    #         trainer = train,
    #         criterion = ,
    #         num_iterations=1,
    #         epochs_per_iteration=1,
    #         pruning_algorithm='fpgm',
    #     )
    # elif opt.pruner == 'taylor':
    #     config_list = [{'sparsity': args.sparsity, 'op_types': ['Conv2d']}]
    #     pruner = TaylorFOWeightFilterPruner(
    #         model,
    #         config_list,
    #         optimizer,
    #         trainer,
    #         criterion,
    #         sparsifying_training_batches=1,
    #     )
    elif opt.pruner == 'fpgm':
        pruner = FPGMPruner(model, config_list)
        print('Prune type：FPGMPruner')

    _, masks = pruner.compress()
    pruner.unwrap_model()

    model_speedup = ModelSpeedup(model, dummy_input, masks)
    model_speedup.logger.setLevel(logging.WARNING)
    pruned_model = model_speedup.speedup_model()
    # print(model)
    
    # 保存剪枝后的模型
    container_model_path = ROOT / f"{Path(opt.cfg).stem.split('_')[0]}.pt"
    print(f"Loading {container_model_path = }")
    ckpt = torch.load(container_model_path)
    ckpt["model"] = deepcopy(de_parallel(model))
    ckpt["date"] = None

    output_model_path = output_dir / f'pruned_{Path(opt.cfg).stem}_{opt.pruner}s_{config_list[0]["sparse_ratio"]}.pt'
    print(f"Saving pruned model to {output_model_path}")
    torch.save(ckpt, output_model_path)
    print('Pruned model paramater number: ', sum([param.numel() for param in model.parameters()]))
    pruned_model_size = os.path.getsize(output_model_path) / (1024 * 1024)  # 将字节转换为MB
    print(f"剪枝之后模型 {output_model_path} 的存储占用大小: {pruned_model_size:.2f} MB")

    val_outputs = get_val_result(weights=output_model_path, device=opt.device)
    print(f"{val_outputs = }")

