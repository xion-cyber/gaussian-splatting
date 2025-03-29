# !/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time     : 12/10/24 1:28 PM
# @Author   : YeYiqi
# @Email    : yeyiqi@stu.pku.edu.cn
# @File     : cameras_from_colmap.py
# @desc     :

import numpy as np

def parse_cameras(intrinsics_file_path, extrinsics_file_path):
    cameras = {}
    with open(intrinsics_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            camera_id = f'{int(parts[0])-1:02}'
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            focal = [float(parts[4]), float(parts[5])]
            principal = [float(parts[6]), float(parts[7])]
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'K': np.array([
                    [focal[0], 0, principal[0]],
                    [0, focal[1], principal[1]],
                    [0, 0, 1]
                ])
            }
    with open(extrinsics_file_path, 'r') as f:
        lines = f.readlines()
        flag = True
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            if flag:
                parts = line.strip().split()
                camera_id = f'{int(parts[0])-1:02}'
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                cameras[camera_id]['R'] = np.array([
                    [1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                    [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
                    [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]
                ])
                cameras[camera_id]['T'] = np.array([tx, ty, tz])
                flag = False
            else:
                flag = True

    output_file_path = './annots.npy'
    np.save(output_file_path, cameras)
    print(f"Cameras saved to {output_file_path}")

if __name__ == '__main__':
    intrinsics_file_path = '/home/yeyiqi/Documents/dataset/GeneBody/fuzhizhi/image/colmap/cameras.txt'
    extrinsics_file_path = '/home/yeyiqi/Documents/dataset/GeneBody/fuzhizhi/image/colmap/images.txt'
    parse_cameras(intrinsics_file_path, extrinsics_file_path)


