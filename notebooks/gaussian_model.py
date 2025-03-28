# !/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time     : 3/12/25 11:23 AM
# @Author   : YeYiqi
# @Email    : yeyiqi@stu.pku.edu.cn
# @File     : gaussian_model.py.py
# @desc     :


from typing import Optional
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
# from pytorch3d.transforms import quaternion_multiply
from roma import quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

