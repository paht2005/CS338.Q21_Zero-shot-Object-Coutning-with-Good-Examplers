#!/usr/bin/env python3
"""
Modified FSC_test.py that saves per-image results to CSV.

Run from code/source-code/ with CUDA:
  python ../../scripts/generate_per_image_csv.py \
      --resume ./data/checkpoint_FSC.pth \
      --anno_file annotation_FSC147_pos.json \
      --anno_file_negative ./data/FSC147/annotation_FSC147_neg.json \
      --output_csv ../../docs/report/figures/per_image_baseline.csv

Repeat for each configuration:
  1. VA-Count:          --resume data/checkpoint_FSC.pth
                        --anno_file annotation_FSC147_pos.json
  2. +Rich Prompt:      --resume data/checkpoint__finetuning_dino_prompt.pth
                        --anno_file annotation_FSC147_pos_prompt.json
  3. +YOLO-World:       --resume data/checkpoint__finetuning_yolo_noprompt.pth
                        --anno_file annotation_FSC147_pos_yolo.json
  4. +YOLO+RP:          --resume data/checkpoint__finetuning_yolo.pth
                        --anno_file annotation_FSC147_pos_yolo_prompt.json
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import timm
from tqdm import tqdm

import util.misc as misc
import models_mae_cross
from util.FSC147 import transform_train, transform_val

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str)
    parser.add_argument('--norm_pix_loss', action='store_true')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--data_path', default='./data/FSC147/', type=str)
    parser.add_argument('--anno_file', default='annotation_FSC147_pos.json', type=str)
    parser.add_argument('--anno_file_negative', default='./data/FSC147/annotation_FSC147_neg.json', type=str)
    parser.add_argument('--data_split_file', default='Train_Test_Val_FSC_147.json', type=str)
    parser.add_argument('--im_dir', default='images_384_VarV2', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='./data/checkpoint_FSC.pth', type=str)
    parser.add_argument('--external', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--output_csv', default='per_image_results.csv', type=str,
                        help='Path to save per-image CSV results')
    return parser


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load data
    data_path = Path(args.data_path)
    anno_file = data_path / args.anno_file
    data_split_file = data_path / args.data_split_file

    with open(anno_file) as f:
        annotations = json.load(f)
    with open(data_split_file) as f:
        data_split = json.load(f)

    test_imgs = data_split['test']

    # Build dataset (reuse existing dataset class)
    from util.FSC147 import TestData
    dataset_test = TestData(
        data_path=str(data_path),
        anno_file=str(anno_file),
        anno_file_negative=args.anno_file_negative,
        data_split_file=str(data_split_file),
        im_dir=args.im_dir,
        split='test',
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Load model
    model = models_mae_cross.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.to(device)
    misc.load_model_FSC(args=args, model_without_ddp=model)
    model.eval()

    # Run inference and collect per-image results
    results = []
    idx = 0
    for val_samples, val_gt_density, val_n_ppl, val_boxes, neg_val_boxes, val_pos, _, val_im_names in tqdm(data_loader_test, desc="Test"):
        val_samples = val_samples.to(device, non_blocking=True, dtype=torch.float)
        val_gt_density = val_gt_density.to(device, non_blocking=True, dtype=torch.float)
        val_boxes = val_boxes.to(device, non_blocking=True, dtype=torch.float)
        neg_val_boxes = neg_val_boxes.to(device, non_blocking=True, dtype=torch.float)

        actual_shot_num = min(5, val_boxes.shape[1])

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                output = model(val_samples, val_boxes, actual_shot_num)

        pred_count = torch.abs(output.sum()).item() / 60.0
        gt_count = val_gt_density.sum().item() / 60.0
        mae = abs(pred_count - gt_count)

        # Get image name
        img_name = val_im_names[0] if isinstance(val_im_names, (list, tuple)) else val_im_names

        # Get density and object size from annotations
        entry = annotations.get(img_name, {})
        h = entry.get('H', 1)
        w = entry.get('W', 1)
        gt_pts = len(entry.get('points', []))
        density = gt_pts / (h * w) * 1e6

        boxes_raw = entry.get('box_examples_coordinates', [])
        areas = []
        for box in boxes_raw:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            bw = max(xs) - min(xs)
            bh = max(ys) - min(ys)
            areas.append(bw * bh)
        avg_size = sum(areas) / len(areas) if areas else 0

        results.append({
            'image': img_name,
            'gt_count': round(gt_count, 2),
            'pred_count': round(pred_count, 2),
            'mae': round(mae, 2),
            'density': round(density, 2),
            'avg_obj_size': round(avg_size, 2),
        })
        idx += 1

    # Save CSV
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    total_mae = sum(r['mae'] for r in results) / len(results)
    print(f'Saved {len(results)} rows to {args.output_csv}')
    print(f'Overall MAE: {total_mae:.2f}')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
