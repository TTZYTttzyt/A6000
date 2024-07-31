# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from typing import Optional

import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
import json
from omni.isaac.sensor import Camera


class CameraData(Camera):
    def __init__(
        self,
        prim_paths_expr: str,
        position: Optional[torch.tensor] = None, #np.array([7.0,0.11,0.8]),
        resolution:Optional[tuple] = None, #=(256, 256),
        orientation:Optional[torch.tensor] = None, #= torch.tensor([0,0,0,1])
        name: Optional[str] = "camera",
    ) -> None:
        self._position = position
        self._resolution = resolution
        self._orientation = orientation
        self._name = name


        super().__init__(prim_paths_expr=prim_paths_expr, 
                         name=name, 
                         position=position,
                         resolution=resolution,
                         orientation=orientation)





# 定义自定义的JSON编码器
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class MobilityAnnotation(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "mobility_annotation",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]"""
        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
        self._usd_path = "/home/lwh/lhr/zyt/40147/mobility_annotation_gapartnet/mobility_annotation_gapartnet.usd" #"/home/lwh/lhr/zyt/mobility/mobility.usd"
            #assets_root_path +  "/Isaac/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd"  mobility/

        add_reference_to_stage(self._usd_path, prim_path)

        self._position = torch.tensor([0.2389, 0.297, 0.4913]) if translation is None else translation
        self._orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation
        
        # # 绕 Z 轴旋转 180 度（π）
        # rotation_angle = torch.tensor([0.0, 0.0, 1.0, 0.0]) * torch.tensor([0.0, 0.0, 0.0, 1.0])

        # # 使用旋转角度初始化朝向，如果朝向未提供，则使用默认旋转角度
        # self._orientation = rotation_angle #if orientation is None else orientation



        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )
