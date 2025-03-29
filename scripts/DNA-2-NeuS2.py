# !/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time     : 3/26/25 12:03 PM
# @Author   : YeYiqi
# @Email    : yeyiqi@stu.pku.edu.cn
# @File     : DNA-2-NeuS2.py
# @desc     :

import os
import argparse
from PIL import Image
import glob
import numpy as np
import json


def process_images(input_dir, frame_base, output_dir="result"):
    """
    处理多视角图像和掩码，生成带透明通道的合成图像
    :param input_dir: 包含img和masks文件夹的根目录
    :param frame_base: 要处理的帧名称基础部分（如"frame_0001"）
    :param output_dir: 输出目录
    """
    img_root = os.path.join(input_dir, "img")
    mask_root = os.path.join(input_dir, "refined_mask")
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有视图编号
    views = [d for d in os.listdir(img_root)
             if os.path.isdir(os.path.join(img_root, d))]

    for view in views:
        # 构建视图路径
        img_view_dir = os.path.join(img_root, view)
        mask_view_dir = os.path.join(mask_root, view)

        # 查找匹配的图像文件
        img_files = glob.glob(os.path.join(img_view_dir, f"{frame_base}.*"))
        img_files = [f for f in img_files
                     if os.path.splitext(f)[1].lower() in ('.png', '.jpg', '.jpeg')]

        # 查找匹配的掩码文件
        mask_files = glob.glob(os.path.join(mask_view_dir, f"{frame_base}.*"))
        mask_files = [f for f in mask_files
                      if os.path.splitext(f)[1].lower() in ('.png', '.jpg', '.jpeg')]

        if not img_files or not mask_files:
            print(f"视图 {view} 缺少图像或掩码，跳过处理")
            continue

        # 使用找到的第一个匹配文件
        img_path = img_files[0]
        mask_path = mask_files[0]

        try:
            # 打开图像并转换为RGB
            img = Image.open(img_path).convert("RGB")

            # 打开掩码并转换为二值化灰度图
            mask = Image.open(mask_path).convert("L")
            mask = mask.point(lambda x: 255 if x > 128 else 0)  # 二值化处理

            # 创建带透明通道的图像
            rgba = Image.new("RGBA", img.size)
            rgba.paste(img, (0, 0), mask=mask)


            # 保存结果（保持原始文件名）
            output_name = f"{img_path.split('/')[-2]}.png"
            output_path = os.path.join(output_dir, "images", output_name)
            rgba.save(output_path, "PNG")
            print(f"成功处理视图 {view} -> {output_path}")

        except Exception as e:
            print(f"处理视图 {view} 失败: {str(e)}")

def save_cam_as_transform(input_dir, output_dir):
    cam_info = np.load(os.path.join(input_dir, "annots.npy"), allow_pickle=True).item()
    frames = []
    for key in cam_info.keys():
        if int(key) > 47:
            continue
        file_path = f"images/{key}.png"
        extri = cam_info[key]['RT']
        # R = extri[:3, :3]
        # T = extri[:3, 3]
        # R_inv = R.T
        # T_inv = -R_inv @ T
        # extri[:3, :3] = R_inv
        # extri[:3, 3] = T_inv
        convert = np.array([[1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]])
        extri = extri @ convert
        intri = cam_info[key]['K']
        intri = np.concatenate(
            (np.concatenate((intri, np.array([[0], [0], [0]])), axis=1), np.array([[0, 0, 0, 1]])), axis=0)
        frame = {
            "file_path": file_path,
            "transform_matrix": extri.tolist(),
            "intrinsic_matrix": intri.tolist()
        }
        frames.append(frame)
    transform = {
        "w": 2048,
        "h": 2448,
        "aabb_scale": 2,
        "scale": 0.33,
        "offset": [0.5, 0.5, 0.5],
        "frames": frames
    }
    with open(os.path.join(output_dir,"transform.json"), "w") as f:
        json.dump(transform, f)
    with open(os.path.join(output_dir,"transform_train.json"), "w") as f:
        json.dump(transform, f)
    with open(os.path.join(output_dir,"transform_test.json"), "w") as f:
        json.dump(transform, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多视角图像掩码合成工具")
    parser.add_argument("--input", required=True, help="包含img和masks文件夹的根目录")
    parser.add_argument("--frame", required=True, help="要处理的帧名称基础部分（如frame_0001）")
    parser.add_argument("--output", default="result", help="输出目录（默认：result）")

    args = parser.parse_args()

    process_images(
        input_dir=args.input,
        frame_base=args.frame,
        output_dir=args.output
    )

    save_cam_as_transform(args.input, args.output)