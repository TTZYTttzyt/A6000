# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import math

import numpy as np
import torch
from omni.isaac.cloner import Cloner
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.torch.transformations import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.mobility_annotation import MobilityAnnotation,NumpyEncoder,CameraData
from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.views.mobility_annotation_view import MobilityAnnotationView
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView
# from omni.replicator.isaac.scripts.writers.pytorch_listener import PytorchListener
from omni.isaac.core.utils.semantics import add_update_semantics
        
import omni.isaac.core.utils.numpy.rotations as rot_utils
from pxr import Usd, UsdGeom
import json
import os
import omni
# from urdfpy import URDF
import omni.isaac.core.utils.prims as prims_utils
from pxr import Gf
import sys
sys.path.append('/home/lwh/.local/share/ov/pkg/isaac_sim-2023.1.1/exts') #/exts/omni.isaac.sensor')
from omni.isaac.sensor import Camera


class FrankaAnnotationMobilityTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self.distX_offset = 0.04
        self.dt = 1 / 60.0

        self._num_observations = self.camera_width * self.camera_height * 2 #22  #23
        self._num_actions = 9

        RLTask.__init__(self, name, env)
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]
        self.num_props = self._task_cfg["env"]["numProps"]
        self.camera_width = self._task_cfg["env"]["cameraWidth"]
        self.camera_height = self._task_cfg["env"]["cameraHeight"]
        

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
            (self.num_envs, self.camera_width, self.camera_height, 2), device=self.device, dtype=torch.float)


    def set_up_scene(self, scene) -> None:
        self.get_franka()
        self.get_mobility()
        # self.add_camera()
        if self.num_props > 0:
            self.get_props()

        super().set_up_scene(scene, filter_collisions=False)

        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self._mobilitys = MobilityAnnotationView(prim_paths_expr="/World/envs/.*/mobility_annotation", name="mobility_view")

        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self._mobilitys)
        scene.add(self._mobilitys._drawers)

        if self.num_props > 0:
            self._props = RigidPrimView(
                prim_paths_expr="/World/envs/.*/prop/.*", name="prop_view", reset_xform_properties=False
            )
            scene.add(self._props)

        camera = Camera(
        prim_path=self.default_zero_env_path + "/camera", #f"/World/envs/env_0/camera",
        position=np.array([7.0,0.11,0.8]),
        # frequency=60, #20,
        resolution=(self.camera_width, self.camera_height),
        orientation= torch.tensor([0,0,0,1]) #rot_utils.euler_angles_to_quats(np.array([0.0,90.0,0.0]), degrees=True),  0.254,-0.65,0.226,0.679
        )
        self._sim_config.apply_articulation_settings(
            "camera", get_prim_at_path(camera.prim_path), self._sim_config.parse_actor_config("camera")
        )
        camera.initialize()

        camera.add_semantic_segmentation_to_frame()
        camera.add_distance_to_image_plane_to_frame()
        self.camera = camera
        
        # for key, value in camera.get_current_frame()["semantic_segmentation"].items():
        #     print(f"Key: {key}")
            
        #     if isinstance(value, np.ndarray):
        #         print("Type: NumPy Array")
        #         print("Values:", value)
        #         print("Shape:", value.shape)
        #         print("Number of elements:", value.size)


        stage = get_current_stage()
        for i in range (self._num_envs):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/mobility_annotation/link_4")
            add_update_semantics(prim, semantic_label="handle", type_label="class")

        self.init_data()
        return

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("franka_view"):
            scene.remove_object("franka_view", registry_only=True)
        if scene.object_exists("hands_view"):
            scene.remove_object("hands_view", registry_only=True)
        if scene.object_exists("lfingers_view"):
            scene.remove_object("lfingers_view", registry_only=True)
        if scene.object_exists("rfingers_view"):
            scene.remove_object("rfingers_view", registry_only=True)
        if scene.object_exists("mobility_annotation_view"):
            scene.remove_object("mobility_annotation_view", registry_only=True)
        if scene.object_exists("drawers_view"):
            scene.remove_object("drawers_view", registry_only=True)
        if scene.object_exists("prop_view"):
            scene.remove_object("prop_view", registry_only=True)
        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self._mobilitys = MobilityAnnotationView(prim_paths_expr="/World/envs/.*/mobility_annotation", name="mobility_view")

        
        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self._mobilitys)
        scene.add(self._mobilitys._drawers)

        if self.num_props > 0:
            self._props = RigidPrimView(
                prim_paths_expr="/World/envs/.*/prop/.*", name="prop_view", reset_xform_properties=False
            )
            scene.add(self._props)

        self.init_data()

    def get_franka(self):
        franka = Franka(prim_path=self.default_zero_env_path + "/franka", name="franka")
        self._sim_config.apply_articulation_settings(
            "franka", get_prim_at_path(franka.prim_path), self._sim_config.parse_actor_config("franka")
        )

    

    def get_mobility(self):
        mobility_annotation = MobilityAnnotation(self.default_zero_env_path + "/mobility_annotation", name="mobility_annotation")
        self._sim_config.apply_articulation_settings(
            "mobility_annotation", get_prim_at_path(mobility_annotation.prim_path), self._sim_config.parse_actor_config("mobility_annotation")
        )
        


    def add_camera(self) -> None:
        camera = Camera(
        prim_path=self.default_zero_env_path + "/camera", #f"/World/envs/env_0/camera",
        position=np.array([7.0,0.11,0.8]),
        # frequency=60, #20,
        resolution=(256, 256),
        orientation= torch.tensor([0,0,0,1]) #rot_utils.euler_angles_to_quats(np.array([0.0,90.0,0.0]), degrees=True),  0.254,-0.65,0.226,0.679
        )
        self._sim_config.apply_articulation_settings(
            "camera", get_prim_at_path(camera.prim_path), self._sim_config.parse_actor_config("camera")
        )
        camera.initialize()

        camera.add_semantic_segmentation_to_frame()
        camera.add_distance_to_image_plane_to_frame()

        anno_json = "/home/lwh/lhr/zyt/40147/test.json"

        # 将字典保存到JSON文件
        with open(anno_json, 'w', encoding='utf-8') as f:
            json.dump(camera.get_current_frame()["semantic_segmentation"], f, cls=NumpyEncoder, ensure_ascii=False, indent=4)

        print(camera.get_current_frame()["semantic_segmentation"])  #motion_vectors
        for key, value in camera.get_current_frame()["semantic_segmentation"].items():
            print(f"Key: {key}")
            
            if isinstance(value, np.ndarray):
                print("Type: NumPy Array")
                print("Values:", value)
                print("Shape:", value.shape)
                print("Number of elements:", value.size)


    def get_props(self):
        prop_cloner = Cloner()
        drawer_pos = torch.tensor([0.0515, 0.0, 0.7172])
        prop_color = torch.tensor([0.2, 0.4, 0.6])

        props_per_row = int(math.ceil(math.sqrt(self.num_props)))
        prop_size = 0.08
        prop_spacing = 0.09
        xmin = -0.5 * prop_spacing * (props_per_row - 1)
        zmin = -0.5 * prop_spacing * (props_per_row - 1)
        prop_count = 0

        prop_pos = []
        for j in range(props_per_row):
            prop_up = zmin + j * prop_spacing
            for k in range(props_per_row):
                if prop_count >= self.num_props:
                    break
                propx = xmin + k * prop_spacing
                prop_pos.append([propx, prop_up, 0.0])

                prop_count += 1

        prop = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/prop/prop_0",
            name="prop",
            color=prop_color,
            size=prop_size,
            density=100.0,
        )
        self._sim_config.apply_articulation_settings(
            "prop", get_prim_at_path(prop.prim_path), self._sim_config.parse_actor_config("prop")
        )

        prop_paths = [f"{self.default_zero_env_path}/prop/prop_{j}" for j in range(self.num_props)]
        prop_cloner.clone(
            source_prim_path=self.default_zero_env_path + "/prop/prop_0",
            prim_paths=prop_paths,
            positions=np.array(prop_pos) + drawer_pos.numpy(),
            replicate_physics=False,
        )

    def init_data(self) -> None:
        def get_env_local_pose(env_pos, xformable, device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_link7")),
            self._device,
        )
        lfinger_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_leftfinger")),
            self._device,
        )
        rfinger_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_rightfinger")),
            self._device,
        )

        finger_pose = torch.zeros(7, device=self._device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        grasp_pose_axis = 1
        franka_local_grasp_pose_rot, franka_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        franka_local_pose_pos += torch.tensor([0, 0.04, 0], device=self._device)
        self.franka_local_grasp_pos = franka_local_pose_pos.repeat((self._num_envs, 1))
        self.franka_local_grasp_rot = franka_local_grasp_pose_rot.repeat((self._num_envs, 1))

        # drawer_local_grasp_pose = torch.tensor([-0.37279,0.45137,0.43318, 1.0, 0.0, 0.0, 0.0], device=self._device)
        # drawer_local_grasp_pose = torch.tensor([0.32315 , 0.32064 ,-0.2979, 0.5, 0.5, 0.5,0.5], device=self._device)
        
        # drawer_local_grasp_pose = torch.tensor([-0.2979  , 0.32315 , 0.32064, 0.5, -0.5,-0.5 ,-0.5], device=self._device) 
        # -0.29821   0.34625    -0.280
        # drawer_local_grasp_pose = torch.tensor([-0.295 , 0.36535 , 0.34925, 0.5, -0.5,-0.5 ,-0.5], device=self._device)
        # -0.28314  0.37739  0.3833
        # drawer_local_grasp_pose = torch.tensor([-0.283 , 0.365 , 0.383, 0.5, -0.5,-0.5 ,-0.5], device=self._device)
        drawer_local_grasp_pose = torch.tensor([-0.283 , 0.382 , 0.383, 0.5, -0.5,-0.5 ,-0.5], device=self._device)


        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self._num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self._num_envs, 1))


        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.drawer_up_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )

        self.franka_default_dof_pos = torch.tensor(
            [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self._device
        )

        hand_pos, hand_rot = self._frankas._hands.get_world_poses(clone=False)
        drawer_pos, drawer_rot = self._mobilitys._drawers.get_world_poses(clone=False)
        (
            self.init_franka_grasp_rot,
            self.init_franka_grasp_pos,
            self.init_drawer_grasp_rot,
            self.init_drawer_grasp_pos,
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

        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)


    def preprocess_segmentation_data(segmentation_data):
        """
        Preprocess segmentation data to convert it into a usable format for training.

        Args:
            segmentation_data (dict): The raw segmentation data dictionary.

        Returns:
            np.ndarray: Preprocessed segmentation image.
        """
        # Extract the actual segmentation image data from the dictionary
        seg_image = np.array(segmentation_data["data"]).reshape((240, 160))  # Assuming the shape is 240x160

        return seg_image

    

    def get_observations(self) -> dict:
        self.segmentation_data = self.camera.get_current_frame()["semantic_segmentation"]
        self.depth_data = self.camera.get_current_frame()["distance_to_image_plane"]

        seg_image = self.preprocess_segmentation_data(self.segmentation_data)
        
        # 将语义分割图像和深度数据组合成obs
        obs = np.stack((seg_image, self.depth_data), axis=2)  # 组合成 (240, 160, 2) 的形状

        anno_json = "/home/lwh/lhr/zyt/40147/test.json"

        # 将字典保存到JSON文件
        with open(anno_json, 'w', encoding='utf-8') as f:
            json.dump(self.segmentation_data, f, cls=NumpyEncoder, ensure_ascii=False, indent=4)

        print("self.segmentation_data",self.segmentation_data)  #motion_vectors
        print("self.depth_data",self.depth_data)
        hand_pos, hand_rot = self._frankas._hands.get_world_poses(clone=False)
        drawer_pos, drawer_rot = self._mobilitys._drawers.get_world_poses(clone=False)
        cabinet_pos,cabinet_rot = self._mobilitys.get_world_poses(clone=False)
        print("cabinet_pos:",cabinet_pos)
        print("cabinet_rot:",cabinet_rot)
        print("................................................................")
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)
        self.mobility_dof_pos = self._mobilitys.get_joint_positions(clone=False)
        self.mobility_dof_vel = self._mobilitys.get_joint_velocities(clone=False)
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
        self.franka_rfinger_pos, self.franka_rfinger_rot = self._frankas._rfingers.get_world_poses(clone=False)

        dof_pos_scaled = (
            2.0
            * (franka_dof_pos - self.franka_dof_lower_limits)
            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
            - 1.0
        )
        
        to_target = self.drawer_grasp_pos - self.franka_grasp_pos
        delta_x = self.drawer_grasp_pos[:,0]-self.init_drawer_grasp_pos[:,0]
        print("delt_x:", delta_x)
        # 9 9 3 1 1
        #9 9 
        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,
                franka_dof_vel * self.dof_vel_scale,
                to_target,
                delta_x.unsqueeze(-1),
                # self.mobility_dof_pos[:, 1].unsqueeze(-1),
                # self.mobility_dof_vel[:, 1].unsqueeze(-1),
                # self.mobility_dof_pos[:, 1].unsqueeze(-1),
                # self.mobility_dof_vel[:, 1].unsqueeze(-1),
            ),
            dim=-1,
        )

        observations = {self._frankas.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        targets = self.franka_dof_targets + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:] = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)

        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # reset franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self._device) - 0.5),
            self.franka_dof_lower_limits,
            self.franka_dof_upper_limits,
        )
        dof_pos = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.franka_dof_targets[env_ids, :] = pos
        self.franka_dof_pos[env_ids, :] = pos

        # reset mobility_annotation
        self._mobilitys.set_joint_positions(
            torch.zeros_like(self._mobilitys.get_joint_positions(clone=False)[env_ids]), indices=indices
        )
        self._mobilitys.set_joint_velocities(
            torch.zeros_like(self._mobilitys.get_joint_velocities(clone=False)[env_ids]), indices=indices
        )

        # reset props
        if self.num_props > 0:
            self._props.set_world_poses(
                self.default_prop_pos[self.prop_indices[env_ids].flatten()],
                self.default_prop_rot[self.prop_indices[env_ids].flatten()],
                self.prop_indices[env_ids].flatten().to(torch.int32),
            )

        self._frankas.set_joint_position_targets(self.franka_dof_targets[env_ids], indices=indices)
        self._frankas.set_joint_positions(dof_pos, indices=indices)
        self._frankas.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):

        self.num_franka_dofs = self._frankas.num_dof
        self.franka_dof_pos = torch.zeros((self.num_envs, self.num_franka_dofs), device=self._device)
        dof_limits = self._frankas.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[self._frankas.gripper_indices] = 0.1
        self.franka_dof_targets = torch.zeros(
            (self._num_envs, self.num_franka_dofs), dtype=torch.float, device=self._device
        )

        if self.num_props > 0:
            self.default_prop_pos, self.default_prop_rot = self._props.get_world_poses()
            self.prop_indices = torch.arange(self._num_envs * self.num_props, device=self._device).view(
                self._num_envs, self.num_props
            )

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = self.compute_franka_reward(
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.mobility_dof_pos,
            self.franka_grasp_pos,
            self.drawer_grasp_pos,
            self.franka_grasp_rot,
            self.drawer_grasp_rot,
            self.franka_lfinger_pos,
            self.franka_rfinger_pos,
            self.gripper_forward_axis,
            self.drawer_inward_axis,
            self.gripper_up_axis,
            self.drawer_up_axis,
            self._num_envs,
            self.dist_reward_scale,
            self.rot_reward_scale,
            self.around_handle_reward_scale,
            self.open_reward_scale,
            self.finger_dist_reward_scale,
            self.action_penalty_scale,
            self.distX_offset,
            self._max_episode_length,
            self.franka_dof_pos,
            self.finger_close_reward_scale,
            self.init_drawer_grasp_pos,
        )

    def is_done(self) -> None:
        # reset if drawer is open or max length reached
        self.reset_buf = torch.where(self.mobility_dof_pos[:, 1] > 0.39, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(
            self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf
        )
        # print("-----------------------------------------------------------------------")

    def compute_grasp_transforms(
        self,
        hand_rot,

        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):

        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )
        # print("global_drawer_rot:",global_drawer_rot)
        global_drawer_rot=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device='cuda:0')
        # print("global_drawer_rot:",global_drawer_rot)
        # print("global_drawer_pos:",global_drawer_pos)
        # print("global_drawer_rot:",global_drawer_rot)
        
        return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos

    def compute_franka_reward(
        self,
        reset_buf,
        progress_buf,
        actions,
        mobility_dof_pos,
        franka_grasp_pos,
        drawer_grasp_pos,
        franka_grasp_rot,
        drawer_grasp_rot,
        franka_lfinger_pos,
        franka_rfinger_pos,
        gripper_forward_axis,
        drawer_inward_axis,
        gripper_up_axis,
        drawer_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        around_handle_reward_scale,
        open_reward_scale,
        finger_dist_reward_scale,
        action_penalty_scale,
        distX_offset,
        max_episode_length,
        joint_positions,
        finger_close_reward_scale,
        init_drawer_grasp_pos
    ):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float, Tensor) -> Tuple[Tensor, Tensor]

        # 1 distance from hand to the drawer     same
        d = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.05, dist_reward * 2, dist_reward)   #2   0.02
        # 2 rot   #0.5    same
        axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
        axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

        dot1 = (
            torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of forward axis for gripper
        dot2 = (
            torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)    

        #3. weight=5   same
        # reward for distance of each finger from the drawer
        finger_dist_reward = torch.zeros_like(rot_reward)
        lfinger_dist = torch.abs(franka_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        rfinger_dist = torch.abs(franka_rfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        
        finger_dist_reward = torch.where(
            franka_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
            torch.where(
                franka_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
                (0.04 - lfinger_dist) + (0.04 - rfinger_dist),
                finger_dist_reward,
            ),
            finger_dist_reward,
        )

        #4.  weight=0.125    same
        # bonus if left finger is above the drawer handle and right below
        around_handle_reward = torch.zeros_like(rot_reward)
        around_handle_reward = torch.where(
            franka_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
            torch.where(
                franka_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2], around_handle_reward + 1, around_handle_reward
            ),
            around_handle_reward,
        )    
        
        # 5. grasp (close) weight=0.5   same
        d_x=torch.abs(franka_grasp_pos[:,0] - drawer_grasp_pos[:,0])
        is_grasb=d_x<0.01
        finger_close_reward = torch.zeros_like(rot_reward)
        finger_close_reward = torch.where(
            d <= 0.03, (0.04 - joint_positions[:, 7]) + (0.04 - joint_positions[:, 8]), finger_close_reward
        ) * is_grasb * around_handle_reward
        
        
        # 6. how far the mobility_annotation has been opened out    #weight=7.5
        # open_reward = mobility_dof_pos[:, 1] * around_handle_reward + mobility_dof_pos[:, 1]  # drawer_top_joint
        delta_x=drawer_grasp_pos[:,0]-init_drawer_grasp_pos[:,0]
        # open_reward = mobility_dof_pos[:, 1] * around_handle_reward + mobility_dof_pos[:, 1]
        # open_reward_0 = mobility_dof_pos[:, 0] * around_handle_reward + mobility_dof_pos[:, 0]
        open_reward_2=delta_x * (around_handle_reward + 1)
        
        # print("mobility_dof_pos[:, 1]:",mobility_dof_pos[:, 1])
        # print("mobility_dof_pos[:, ]:",mobility_dof_pos)
        # print("init_drawer_grasp_pos",init_drawer_grasp_pos)
        # print("delta_x",delta_x)
        

        # regularization on the actions (summed for each environment)   weight=0.01
        action_penalty = torch.sum(actions**2, dim=-1)


        # print("dist_reward",dist_reward)
        # print("rot_reward",rot_reward)
        # print("franka_lfinger_pos:",franka_lfinger_pos)
        # print("franka_rfinger_pos:",franka_rfinger_pos)
        # print("drawer_grasp_pos:",drawer_grasp_pos)


        # actionScale: 7.5
        # dofVelocityScale: 0.1
        # distRewardScale: 2.0
        # rotRewardScale: 0.5
        # aroundHandleRewardScale: 10.0
        # openRewardScale: 7.5
        # fingerDistRewardScale: 100.0
        # actionPenaltyScale: 0.01
        # fingerCloseRewardScale: 10.0

        # print("around_handle_reward",around_handle_reward)
        # print("open_reward",open_reward)

        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + around_handle_reward_scale * around_handle_reward
            # + open_reward_scale * open_reward
            + open_reward_scale * open_reward_2
            + finger_dist_reward_scale * finger_dist_reward
            - action_penalty_scale * action_penalty
            + finger_close_reward * finger_close_reward_scale
        )

        # print("lfinger_dist:",lfinger_dist)
        # print("rfinger_dist:",rfinger_dist)
        # print("joint_positions[:, 7]:",joint_positions[:, 7])
        # print("joint_positions[:, 8]:",joint_positions[:, 8])
        # print("franka_grasp_pos:",franka_grasp_pos)
        # print("drawer_grasp_pos:",drawer_grasp_pos)
        # print("d",d)
        # print("d_x",d_x)
        # print("dist_reward_scale * dist_reward:",dist_reward_scale * dist_reward)
        # print("rot_reward_scale * rot_reward:",rot_reward_scale * rot_reward)
        # print("finger_dist_reward_scale * finger_dist_reward:",finger_dist_reward_scale * finger_dist_reward)
        # print("around_handle_reward_scale * around_handle_reward:",around_handle_reward_scale * around_handle_reward)
        # print("finger_close_reward * finger_close_reward_scale:",finger_close_reward * finger_close_reward_scale)
        # # print("open_reward_scale * open_reward:",open_reward_scale * open_reward)
        # print("open_reward_scale * open_reward_2:",open_reward_scale * open_reward_2)
        # print("action_penalty_scale * action_penalty:",action_penalty_scale * action_penalty)
        # print("---------------------------------------------------------------------------------")
        

        # bonus for opening drawer properly
        rewards = torch.where(mobility_dof_pos[:, 1] > 0.01, rewards + 0.5, rewards)
        rewards = torch.where(mobility_dof_pos[:, 1] > 0.2, rewards + around_handle_reward, rewards)
        rewards = torch.where(mobility_dof_pos[:, 1] > 0.39, rewards + (2.0 * around_handle_reward), rewards)
        # print("mobility_dof_pos:",mobility_dof_pos)

        # # prevent bad style in opening drawer
        # rewards = torch.where(franka_lfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
        #                       torch.ones_like(rewards) * -1, rewards)
        # rewards = torch.where(franka_rfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
        #                       torch.ones_like(rewards) * -1, rewards)

        return rewards
