# !/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time     : 9/27/24 3:08 PM
# @Author   : YeYiqi
# @Email    : yeyiqi@stu.pku.edu.cn
# @File     : make_masks.py
# @desc     : Generate mask for the input image


from PIL import Image
from pathlib import Path
from transformers import pipeline
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser


def load_images_from_folder(folder_path):
    images = []
    jpg_filenames = folder_path.glob('*.jpg')
    png_filenames = folder_path.glob('*.png')
    filenames = list(jpg_filenames) + list(png_filenames)
    for filename in filenames:
        img = Image.open(filename)
        images.append(img)
    return images, filenames

def boxes_extractor(images, prompt, device, filenames):
    detector = pipeline(
        model = "google/owlv2-base-patch16-ensemble",
        task = "zero-shot-object-detection",
        device=device
    )
    boxes = []
    for i in tqdm(range(len(images)), desc="Extracting boxes"):
        output = detector(
            images[i],
            candidate_labels = [prompt]
        )
        try:
            box = output[0]['box']
            box_np = np.array([box['xmin'], box['ymin'], box['xmax'], box['ymax']])
            boxes.append(box_np)
        except:
            print("{}No box detected".format(filenames[i].name))
            exit(1)
    return boxes

def masks_generator(images, boxes, device):
    sam2_checkpoint = "/home/yeyiqi/Documents/repos/myrepo/segment-anything-2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device)
    predictor = SAM2ImagePredictor(sam2_model)
    masks = []
    for i in tqdm(range(len(images)), desc="Generating masks"):
        predictor.set_image(np.array(images[i].convert("RGB")))
        mask, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes[i][None, :],
            multimask_output=False,
        )
        print(mask.shape)
        masks.append(mask)
    return masks

def save_masks(masks, filenames):
    save_path = filenames[0].parents[0] / "mask"
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    for i in range(len(masks)):
        mask = masks[i]
        mask_uint8 = mask.astype(np.uint8)
        mask_uint8 *= 255
        mask_img = Image.fromarray(mask_uint8.squeeze())
        mask_img.save(save_path / filenames[i].name)



if __name__ == '__main__':
    parser = ArgumentParser(description="Mask generator script parameters")
    parser.add_argument("-input_source", "-s", required=True, type=str, help="Path to the images folder")
    folder_path = Path(parser.parse_args().input_source)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    images, filenames = load_images_from_folder(folder_path)
    prompt = "Dancer"
    boxes = boxes_extractor(images, prompt, device, filenames)
    masks = masks_generator(images, boxes, device)
    save_masks(masks, filenames)

