# !/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time     : 11/27/24 10:57 AM
# @Author   : YeYiqi
# @Email    : yeyiqi@stu.pku.edu.cn
# @File     : make_masks_under_folder.py
# @desc     :

from PIL import Image
from pathlib import Path

from click import prompt
from transformers import pipeline
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import logging


def generate_mask_from_folders(main_folder_path, detector, predictor, prompts):
    main_folder = Path(main_folder_path)
    sub_folders = [sub_folder for sub_folder in main_folder.iterdir() if sub_folder.is_dir()]
    result_folder = main_folder / '..' / 'masks'
    result_folder.mkdir(exist_ok=True)
    for sub_folder in sub_folders:
        result_subfolder = result_folder / sub_folder.name
        result_subfolder.mkdir(exist_ok=True)
        print("Processing folder: {}".format(sub_folder.name))
        images = []
        jpg_filenames = sub_folder.glob('*.jpg')
        png_filenames = sub_folder.glob('*.png')
        filenames = list(jpg_filenames) + list(png_filenames)
        for filename in filenames:
            img = Image.open(filename)
            images.append(img)
        boxes = []
        for i in tqdm(range(len(images)), desc="Extracting boxes"):
            for prompt in prompts:
                output = detector(
                    images[i],
                    candidate_labels = [prompt]
                )
                try:
                    box = output[0]['box']
                    box_np = np.array([box['xmin'], box['ymin'], box['xmax'], box['ymax']])
                    boxes.append(box_np)
                    break
                except Exception as e:
                    logging.error(f"Unexpected error for image {filenames[i].name}: {str(e)} with prompt {prompt}")
                    continue
        masks = []
        for i in tqdm(range(len(images)), desc="Generating masks"):
            predictor.set_image(np.array(images[i].convert("RGB")))
            mask, scores, logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes[i][None, :],
                multimask_output=False

            )
            masks.append(mask)
        for i in tqdm(range(len(masks)), desc="Saving masks"):
            mask = masks[i]
            mask_uint8 = mask.astype(np.uint8)
            mask_uint8 = mask_uint8 * 255
            mask_img = Image.fromarray(mask_uint8.squeeze())
            mask_img.save(result_subfolder / filenames[i].name)



if __name__ == '__main__':
    parser = ArgumentParser(description="Mask generator script parameters")
    parser.add_argument("-folder_path", "-p", required=True, type=str, help="Path to the images folder")
    parser.add_argument("--prompts", nargs='+', type=str, help="Path to the images folder",
                        default=['Dancer', 'Human', 'Person', 'people', 'man', 'woman'])
    args = parser.parse_args()

    logging.basicConfig(
        filename='error_log.txt',
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    detector = pipeline(
        model = "google/owlv2-base-patch16-ensemble",
        task = "zero-shot-object-detection",
        device=device
    )
    sm2_checkpoint = "/home/yeyiqi/Documents/repos/myrepo/segment-anything-2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    sm2_model = build_sam2(model_cfg, sm2_checkpoint, device)
    predictor = SAM2ImagePredictor(sm2_model)
    generate_mask_from_folders(args.folder_path, detector, predictor, args.prompts)
