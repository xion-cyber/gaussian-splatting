# !/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time     : 3/29/25 6:08 PM
# @Author   : YeYiqi
# @Email    : yeyiqi@stu.pku.edu.cn
# @File     : DNA_2_transform.py
# @desc     :

import numpy as np
from pathlib import Path
import json

def create_transform(path: str, test_views:list, vali_views:list):
    """
    Create transform.json file for extracted images of DNA_Rendering dataset
    Args:
        img_root: extracted images root folder containing views
        output: transform.json file path
    """
    path = Path(path)
    img_root = path / 'extracted_img'
    annots_path = path / 'annots.npy'
    annots = np.load(annots_path, allow_pickle=True).item()
    views = [d for d in img_root.iterdir() if d.is_dir()]
    frames = []
    frames_test = []
    frames_val = []
    for view in views:
        view_name = view.name
        img_files = list(view.glob('*.png'))
        if view_name in test_views:
            for img_file in img_files:
                frame = {}
                frame['file_path'] = str(img_file.relative_to(path))
                frame['view'] = view_name
                frame['timestep'] = int(img_file.stem)
                frame['transform_matrix'] = (np.linalg.inv(annots[view_name]['RT'])).tolist()
                frame['intrinsic_matrix'] = annots[view_name]['K'].tolist()
                frames_test.append(frame)
        elif view_name in vali_views:
            for img_file in img_files:
                frame = {}
                frame['file_path'] = str(img_file.relative_to(path))
                frame['view'] = view_name
                frame['timestep'] = int(img_file.stem)
                frame['transform_matrix'] = (np.linalg.inv(annots[view_name]['RT'])).tolist()
                frame['intrinsic_matrix'] = annots[view_name]['K'].tolist()
                frames_val.append(frame)
        else:
            for img_file in img_files:
                frame = {}
                frame['file_path'] = str(img_file.relative_to(path))
                frame['view'] = view_name
                frame['timestep'] = int(img_file.stem)
                frame['transform_matrix'] = (np.linalg.inv(annots[view_name]['RT'])).tolist()
                frame['intrinsic_matrix'] = annots[view_name]['K'].tolist()
                frames.append(frame)
    transform = {
        "w": 2048,
        "h": 2448,
        "frames": frames
    }
    transform_test = {
        "w": 2048,
        "h": 2448,
        "frames": frames_test
    }
    transform_val = {
        "w": 2048,
        "h": 2448,
        "frames": frames_val
    }

    with open(path / 'transform.json', 'w') as f:
        json.dump(transform, f, indent=2)
    print("transform.json file created successfully.")
    with open(path / 'transform_test.json', 'w') as f:
        json.dump(transform_test, f, indent=2)
    print("transform_test.json file created successfully.")
    with open(path / 'transform_val.json', 'w') as f:
        json.dump(transform_val, f, indent=2)
    print("transform_val.json file created successfully.")

if __name__ == "__main__":
    test_views = ['00','05','10','15','20','25','30','35','40','45']
    vali_views = ['02','07','12','17','22','27','32','37','42','47']
    create_transform(
        "/home/yeyiqi/Documents/dataset/DNA-Rendering/4k4d/0023_06",
        test_views,
        vali_views
    )


