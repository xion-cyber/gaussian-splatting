# !/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time     : 3/27/25 11:16 AM
# @Author   : YeYiqi
# @Email    : yeyiqi@stu.pku.edu.cn
# @File     : rotate_mesh.py
# @desc     :

import trimesh
import numpy as np

import open3d as o3d
import numpy as np


def rotate_ply(input_path, output_path):
    """
    对 PLY 网格执行旋转操作：
    1. 先绕 Z 轴旋转 90 度
    2. 再绕 Y 轴旋转 90 度
    """
    # 读取 PLY 文件
    mesh = o3d.io.read_triangle_mesh(input_path)

    # 验证是否成功加载网格
    if not mesh.has_vertices():
        raise ValueError("无法加载有效的网格数据，请检查文件路径")

    # 创建组合旋转矩阵 (注意顺序：先 Z 后 Y)
    angle_z = np.pi / 2  # 90 度转弧度
    angle_y = np.pi / 2

    # 绕 Z 轴旋转矩阵
    R_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                    [np.sin(angle_z), np.cos(angle_z), 0],
                    [0, 0, 1]])

    # 绕 Y 轴旋转矩阵
    R_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                    [0, 1, 0],
                    [-np.sin(angle_y), 0, np.cos(angle_y)]])

    # 组合旋转：先 Z 后 Y（矩阵相乘顺序为 R_total = R_y @ R_z）
    R_total = R_y @ R_z

    # 应用旋转到顶点
    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(mesh.vertices) @ R_total.T  # 矩阵转置因为 Open3D 使用列主序
    )

    # 保存结果
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"旋转后的网格已保存至：{output_path}")




if __name__ == "__main__":
    input_file = "/home/yeyiqi/Documents/dataset/DNA-Rendering/4k4d/0023_06/result/neus.ply"  # 输入 PLY 文件路径
    output_file = "/home/yeyiqi/Documents/dataset/DNA-Rendering/4k4d/0023_06/result/rotated_neus.ply"  # 输出 PLY 文件路径

    rotate_ply(input_file, output_file)