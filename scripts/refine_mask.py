# !/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time     : 11/27/24 10:57 AM
# @Author   : YeYiqi
# @Email    : yeyiqi@stu.pku.edu.cn
# @File     : make_masks_under_folder.py
# @desc     :

from PIL import Image
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import logging

from pathlib import Path
from PIL import Image
from tqdm import tqdm
import concurrent.futures


def process_single_mask(mask_path, sub_folder, result_subfolder, sam_mask_folder):
    """ 处理单个mask的线程任务 """
    try:
        mask = Image.open(mask_path).convert("L")
        sam_mask_path = sam_mask_folder / sub_folder.stem / f"{mask_path.stem}.jpg"
        sam_mask = Image.open(sam_mask_path).convert("L")

        # 二值化处理
        sam_mask = sam_mask.point(lambda x: 255 if x > 128 else 0)

        # 合成新mask
        refined_mask = Image.new("L", mask.size)
        refined_mask.paste(mask, (0, 0), sam_mask)
        refined_mask.save(result_subfolder / mask_path.name)
        return True
    except Exception as e:
        print(f"Error processing {mask_path}: {str(e)}")
        return False


def refine_mask(main_folder_path, max_workers=8):
    """ 多线程版本 """
    main_folder = Path(main_folder_path)
    mask_folder = main_folder / 'mask'
    sam_mask_folder = main_folder / 'masks'
    result_folder = main_folder.parent / 'refined_mask'  # 使用 parent 替代 ..

    # 准备目录结构
    result_folder.mkdir(exist_ok=True)
    sub_folders = [f for f in mask_folder.iterdir() if f.is_dir()]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sub_folder in sub_folders:
            result_subfolder = result_folder / sub_folder.name
            result_subfolder.mkdir(exist_ok=True)

            mask_paths = list(sub_folder.glob('*.png'))
            print(f"Processing {sub_folder.name} ({len(mask_paths)} masks)")

            # 提交任务
            futures = []
            for mask_path in mask_paths:
                future = executor.submit(
                    process_single_mask,
                    mask_path, sub_folder,
                    result_subfolder, sam_mask_folder
                )
                futures.append(future)

            # 进度监控
            with tqdm(total=len(mask_paths), desc=sub_folder.name) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    if future.result():
                        pbar.update(1)


if __name__ == "__main__":
    refine_mask("/home/yeyiqi/Documents/dataset/DNA-Rendering/4k4d/0023_06", max_workers=16)



