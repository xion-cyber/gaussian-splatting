# !/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time     : 3/29/25 4:14 PM
# @Author   : YeYiqi
# @Email    : yeyiqi@stu.pku.edu.cn
# @File     : extract_img_folders.py
# @desc     :
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import concurrent.futures

def extract_single_img(img_path, mask_main_folder, result_subfolder):
    """ Extract single image from folder """
    try:
        mask_path = mask_main_folder / result_subfolder.stem / f"{img_path.stem}.png"
        img =Image.open(img_path).convert("RGBA")
        mask = Image.open(mask_path).convert("L")
        img.putalpha(mask)
        img.save(result_subfolder / f"{img_path.stem}.png")
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return False


def extract_img_from_folders(path: str, output_path: str, max_workers: int = 16):

    """
    Extract images from folders and save them to a new folder
    :param path: the path of the folder containing images and masks
    :param output_path: the path of the folder to save the extracted images
    """
    path = Path(path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    img_main_folder = path / 'img'
    mask_main_folder = path / 'mask'
    sub_folders = [f for f in img_main_folder.iterdir() if f.is_dir()]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sub_folder in sub_folders:
            result_subfolder = output_path / sub_folder.name
            result_subfolder.mkdir(exist_ok=True)
            img_paths = list(sub_folder.glob('*.jpg'))
            print(f"Processing {sub_folder.name} ({len(img_paths)} images)")
            futures = []
            for img_path in img_paths:
                futures.append(executor.submit(extract_single_img, img_path, mask_main_folder, result_subfolder))

            with tqdm(total=len(futures), desc=f"Extracting {sub_folder.name}") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    if future.result():
                        pbar.update(1)

if __name__ == "__main__":
    extract_img_from_folders(
        path="/home/yeyiqi/Documents/dataset/DNA-Rendering/4k4d/0023_06",
        output_path="/home/yeyiqi/Documents/dataset/DNA-Rendering/4k4d/0023_06/extracted_img",
    )





