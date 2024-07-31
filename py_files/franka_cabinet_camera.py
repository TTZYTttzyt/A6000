# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from gym import spaces
import numpy as np
import torch
import omni.usd
from pxr import UsdGeom
import os

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.tasks.franka_cabinet import FrankaCabinetTask
from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.views.cabinet_view import CabinetView
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView
from omni.isaac.core.prims import RigidPrim, RigidPrimView
# from omni.isaac.sensor import Camera


class FrankaCabinetCameraTask(FrankaCabinetTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        print('init')
        self.update_config(sim_config)
        self._max_episode_length = 500

        self.dt = 1 / 60.0
        self.distX_offset = 0.04

        self._num_observations = self.camera_width * self.camera_height * 4 #3
        self._num_actions = 1

        self.refresh=1
        self.dir='/home/lwh/lhr/zyt/collect1'
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        

        # use multi-dimensional observation for camera RGB
        self.observation_space = spaces.Box(
            np.ones((self.camera_width, self.camera_height, 4), dtype=np.float32) * -np.Inf, 
            np.ones((self.camera_width, self.camera_height, 4), dtype=np.float32) * np.Inf)

        RLTask.__init__(self, name, env)

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        #self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])
        self.num_props = self._task_cfg["env"]["numProps"]

        #self._reset_dist = self._task_cfg["env"]["resetDist"]
        #self._max_push_effort = self._task_cfg["env"]["maxEffort"]

        self.camera_type = self._task_cfg["env"].get("cameraType", 'rgb')
        self.camera_width = self._task_cfg["env"]["cameraWidth"]
        self.camera_height = self._task_cfg["env"]["cameraHeight"]
        
        self.camera_channels = 3
        self._export_images = self._task_cfg["env"]["exportImages"]

        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]
        self.num_props = self._task_cfg["env"]["numProps"]

        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self._task_cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self._task_cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self._task_cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.finger_close_reward_scale = self._task_cfg["env"]["fingerCloseRewardScale"]


    def cleanup(self) -> None:
        # initialize remaining buffers
        RLTask.cleanup(self)

        # override observation buffer for camera data
        self.obs_buf = torch.zeros(
            (self.num_envs, self.camera_width, self.camera_height, 3), device=self.device, dtype=torch.float)


    # def add_camera(self) -> None:

    def add_camera(self) -> None:
        stage = get_current_stage()
        camera_path = f"/World/envs/env_0/Camera"
        camera_xform = stage.DefinePrim(f'{camera_path}_Xform', 'Xform')
        # set up transforms for parent and camera prims
        position = (3.1,1.3,0.8) #(0.4,0.0,4.0)     #(3.1,1.3,0.8)   #(-4.2, 0.0, 3.0)
        rotation = (0, -6.1155, 22.2) #(0.0,-90.0,0.0)    #(0, -6.1155, 22.2)
        UsdGeom.Xformable(camera_xform).AddTranslateOp()
        UsdGeom.Xformable(camera_xform).AddRotateXYZOp()
        camera_xform.GetAttribute('xformOp:translate').Set(position)
        camera_xform.GetAttribute('xformOp:rotateXYZ').Set(rotation)
        camera = stage.DefinePrim(f'{camera_path}_Xform/Camera', 'Camera')
        UsdGeom.Xformable(camera).AddRotateXYZOp()
        camera.GetAttribute("xformOp:rotateXYZ").Set((90, 0, 90))
        # set camera properties
        camera.GetAttribute('focalLength').Set(24)
        camera.GetAttribute('focusDistance').Set(400)
        # hide other environments in the background
        camera.GetAttribute("clippingRange").Set((0.01, 20.0))
        # self.add_distance_to_image_plane_to_frame()

      

    def set_up_scene(self, scene) -> None:
        self.get_franka()
        self.get_cabinet()
        self.add_camera()

        
        if self.num_props > 0:
            self.get_props()

        RLTask.set_up_scene(self, scene)
        #super().set_up_scene(scene) #filter_collisions=False)

        # start replicator to capture image data
        self.rep.orchestrator._orchestrator._is_started = True

        # set up cameras
        self.render_products = []
        env_pos = self._env_pos.cpu()
        camera_paths = [f"/World/envs/env_{i}/Camera_Xform/Camera" for i in range(self._num_envs)]
        for i in range(self._num_envs):
            render_product = self.rep.create.render_product(camera_paths[i], resolution=(self.camera_width, self.camera_height))
            # annotator = self.rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
            # annotator.attach([render_product])
            self.render_products.append(render_product)


        # initialize pytorch writer for vectorized collection
        self.pytorch_listener = self.PytorchListener()
        self.pytorch_writer = self.rep.WriterRegistry.get("PytorchWriter")
        self.pytorch_writer.initialize(listener=self.pytorch_listener, device="cuda")
        self.pytorch_writer.attach(self.render_products)


        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self._cabinets = CabinetView(prim_paths_expr="/World/envs/.*/cabinet", name="cabinet_view")

        
        if self.num_props > 0:
            self._props = RigidPrimView(
                prim_paths_expr="/World/envs/.*/prop/.*", name="prop_view", reset_xform_properties=False
            )
            scene.add(self._props)
        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self._cabinets)
        scene.add(self._cabinets._drawers)
        

        self.init_data()

        # self._cartpoles = ArticulationView(
        #     prim_paths_expr="/World/envs/.*/Cartpole", name="cartpole_view", reset_xform_properties=False
        # )
        # scene.add(self._cartpoles)
        return

    def get_observations(self) -> dict:
        hand_pos, hand_rot = self._frankas._hands.get_world_poses(clone=False)
        drawer_pos, drawer_rot = self._cabinets._drawers.get_world_poses(clone=False)
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)
        self.cabinet_dof_pos = self._cabinets.get_joint_positions(clone=False)
        self.cabinet_dof_vel = self._cabinets.get_joint_velocities(clone=False)
        self.franka_dof_pos = franka_dof_pos

        (
            self.franka_grasp_rot,
            self.franka_grasp_pos,
            self.drawer_grasp_rot,
            self.drawer_grasp_pos,
        ) = self.compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.franka_local_grasp_rot,
            self.franka_local_grasp_pos,
            drawer_rot,
            drawer_pos,
            self.drawer_local_grasp_rot,
            self.drawer_local_grasp_pos,
        )

        self.franka_lfinger_pos, self.franka_lfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)
        self.franka_rfinger_pos, self.franka_rfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)

        dof_pos_scaled = (
            2.0
            * (franka_dof_pos - self.franka_dof_lower_limits)
            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
            - 1.0
        )
        to_target = self.drawer_grasp_pos - self.franka_grasp_pos

        # self.get_depth()
        # retrieve RGB data from all render products
        # self.refresh+=1
        images = self.pytorch_listener.get_rgb_data()
        depth = self.pytorch_listener.get_depth_data()
        segmentation = self.pytorch_listener.get_segmentation_data()
        if images is not None:
            if self._export_images:
                from torchvision.utils import save_image, make_grid
                img = images/255
                save_image(make_grid(img, nrows = 2), 'franka_cabinet_export.png')

            rgb_data_normalized = torch.swapaxes(images, 1, 3).float()/255.0
            print("shape1:",rgb_data_normalized.shape)
        else:
            print("Image tensor is NONE!")
            
        # if self.refresh%100:
        #     filename = f"image_{self.refresh}.pt"
        #     file_path = os.path.join(self.dir,filename)
        #     torch.save(images,file_path)
        #     filename1 = f"depth_{self.refresh}.pt"
        #     file_path1 = os.path.join(self.dir,filename1)
        #     torch.save(depth,file_path1)
        # 步骤 1: 替换无穷大值
        # 找到非无穷大值中的最大值
        max_finite = depth[~torch.isinf(depth)].max()
        # 将无穷大值替换为最大非无穷大值加一
        replaced_tensor = torch.where(torch.isinf(depth), torch.full_like(depth, max_finite + 1), depth)
        min_val = replaced_tensor.min()
        max_val = replaced_tensor.max()
        # 归一化
        normalized_tensor = (replaced_tensor - min_val) / (max_val - min_val)

        # 将深度数据维度扩展以匹配 RGB 数据的通道维度
        depth_data_expanded = torch.unsqueeze(normalized_tensor, dim=1)
        depth_data_expanded_exchaged =torch.swapaxes(depth_data_expanded,1,3)
        print("shape:",depth_data_expanded_exchaged.shape)

        # 将 RGB 数据和深度数据拼接在一起
        rgbd_data = torch.cat((rgb_data_normalized, depth_data_expanded_exchaged), dim=3)
        print("shape2:",rgbd_data.shape)

        self.obs_buf = rgbd_data.clone()

        # 找到替换后的张量中的最大值
        # max_value = replaced_tensor.max()
        # print(f"Maximum value excluding infinity is: {max_value.item()}")


        # if images is not None:
        #     if self._export_images:
        #         from torchvision.utils import save_image, make_grid
        #         img = images/255
        #         save_image(make_grid(img, nrows = 2), 'franka_cabinet_export.png')

        #     self.obs_buf = torch.swapaxes(images, 1, 3).clone().float()/255.0
        # else:
        #     print("Image tensor is NONE!")

        return self.obs_buf
