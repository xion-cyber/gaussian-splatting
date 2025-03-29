# !/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time     : 12/13/24 3:37 PM
# @Author   : YeYiqi
# @Email    : yeyiqi@stu.pku.edu.cn
# @File     : shape_from_kpts.py
# @desc     : regress SMPL shape from keypoints

import numpy as np
import torch

from pathlib import Path
from human_body_prior.models.body_model import BodyModel
from human_body_prior.models.ik_engine import IK_Engine

