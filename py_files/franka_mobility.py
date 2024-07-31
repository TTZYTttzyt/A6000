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
from omniisaacgymenvs.robots.articulations.mobility import Mobility
from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.views.mobility_view import MobilityView
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView
from pxr import Usd, UsdGeom
import json
import os
import omni
# from urdfpy import URDF
import omni.isaac.core.utils.prims as prims_utils
from pxr import Gf



# def read_urdf(file_path):
#     robot = URDF.load(file_path)
#     urdf_info = {
#         "robot_name": robot.name,
#         "links": {link.name: {"xyz": link.origin.xyz, "rpy": link.origin.rpy, "obj": [v.filename for v in link.visuals]} for link in robot.links},
#         "joints": {joint.name: joint.type for joint in robot.joints}
#     }
#     return urdf_info

import os
# from os.path import join as pjoin
import xml.etree.ElementTree as ET
# import json
import numpy as np
from PIL import Image
import pickle


# def get_id_category(target_id, id_path):
#     target_id = int(target_id)
#     category = None
#     with open(id_path, 'r') as fd:
#         for line in fd:
#             cat = line.rstrip('\n').split(' ')[0]
#             id = int(line.rstrip('\n').split(' ')[1])
#             if id == target_id:
#                 category = cat
#                 break
#     return category


# def read_urdf(urdf_file):
    
#     tree_urdf = ET.parse(urdf_file)
#     link_dict = {}
#     joint_dict = {}
    
#     robot_name = tree_urdf.getroot().attrib['name']
    
#     num_links = 0
#     for link in tree_urdf.iter('link'):
#         num_links += 1
#         name = link.attrib['name']
#         link_dict[name] = {
#             'xyz': [],
#             'rpy': [],
#             'obj': []
#         }
#         for visual in link.iter('visual'):
#             for origin in visual.iter('origin'):
#                 if 'xyz' in origin.attrib:
#                     link_dict[name]['xyz'].append([float(x) for x in origin.attrib['xyz'].split()])
#                 else:
#                     link_dict[name]['xyz'].append([0, 0, 0])
#                 if 'rpy' in origin.attrib:
#                     link_dict[name]['rpy'].append([float(x) for x in origin.attrib['rpy'].split()])
#                 else:
#                     link_dict[name]['rpy'].append([0, 0, 0])
#             for geometry in visual.iter('geometry'):
#                 for mesh in geometry.iter('mesh'):
#                     link_dict[name]['obj'].append(mesh.attrib['filename'])
#             assert len(link_dict[name]['xyz']) == len(link_dict[name]['rpy']) == len(link_dict[name]['obj']), "Length mismatch"
#         # collision mesh is the same as visual mesh, so we don't need to parse it
    
#     num_joints = 0
#     for joint in tree_urdf.iter('joint'):
#         num_joints += 1
#         name = joint.attrib['name']
#         joint_dict[name] = {}



class FrankaMobilityTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self.distX_offset = 0.04
        self.dt = 1 / 60.0

        self._num_observations = 22  #23
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

        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self._task_cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self._task_cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self._task_cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.finger_close_reward_scale = self._task_cfg["env"]["fingerCloseRewardScale"]

    def set_up_scene(self, scene) -> None:
        self.get_franka()
        self.get_mobility()
        if self.num_props > 0:
            self.get_props()

        super().set_up_scene(scene, filter_collisions=False)

        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self._mobilitys = MobilityView(prim_paths_expr="/World/envs/.*/mobility", name="mobility_view")

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
        if scene.object_exists("mobility_view"):
            scene.remove_object("mobility_view", registry_only=True)
        if scene.object_exists("drawers_view"):
            scene.remove_object("drawers_view", registry_only=True)
        if scene.object_exists("prop_view"):
            scene.remove_object("prop_view", registry_only=True)
        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self._mobilitys = MobilityView(prim_paths_expr="/World/envs/.*/mobility", name="mobility_view")

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

    


    def generate_CAD_model_list(model_folder, model_id):
    
        model_path = os.path.join(model_folder, model_id) #pjoin(model_folder, model_id)  # 试试直接用urdf路径传入吧
        
        CAD_model_part_class_link_list = {}
        anno_json = os.path.join(model_path, 'link_annotation_gapartnet.json')
        with open(anno_json, 'r') as f:
            anno_list = json.load(f)
        link_anno_dict = {}
        for link_dict in anno_list:
            link_name = link_dict['link_name']
            is_gapart = link_dict['is_gapart']
            if is_gapart:
                part_class = link_dict['category']
            else:
                part_class = 'other'
            if part_class not in CAD_model_part_class_link_list:
                CAD_model_part_class_link_list[part_class] = [link_name]
            else:
                CAD_model_part_class_link_list[part_class].append(link_name)
            link_anno_dict[link_name] = {
                'is_gapart': is_gapart,
                'part_class': part_class,
                'bbox': np.array(link_dict['bbox'], dtype=np.float32).reshape(-1, 3)
            }

        CAD_model_link_name_mesh_list = {}
        urdf_path = os.path.join(model_path, 'mobility_annotation_gapartnet.urdf')
        urdf_info = read_urdf(urdf_path)     # get information from urdf
        
        link_info = urdf_info['links']
        link_name_list = list(link_info.keys())
        for link_name in link_name_list:
            CAD_model_link_name_mesh_list[link_name] = link_info[link_name]['obj']
        
        link_trans_dict = {}
        link_trans_dict['object_name'] = urdf_info['robot_name']
        for link_name in link_name_list:
            xyzs = link_info[link_name]['xyz']
            rpys = link_info[link_name]['rpy']
            objs = link_info[link_name]['obj']
            for xyz, rpy, obj in zip(xyzs, rpys, objs):
                name = os.path.basename(obj).split(".obj")[0]
                link_trans_dict[name] = [xyz, rpy]  # original-4.obj: [xyz, rpy]
        
        # the first one introduce every link's part class (link-level)
        # the second one introduce the mesh name and path of every link (include the object in each link)
        return CAD_model_part_class_link_list, CAD_model_link_name_mesh_list, link_trans_dict, urdf_info['joints'], link_anno_dict



    def load_usd(CAD_model_link_name_mesh_list, CAD_model_part_class_link_list, link_trans_dic, cad_model_dir, CAD_MODEL, my_world, dr, need_change_material=True, height=0):
    # only one material is set for each type of part, just to look good
        # if need_change_material: 
        #     for part in part_material_pairs.keys(): # part_material_pairs[part]: ['specular', 'metal', 'diffuse', 'transparent', 'glass']
        #         part_material_pairs[part] = random.choice(part_material_pairs[part]) # part_material_pairs[part]: 'specular'
        link_trans_dic['object_name'] = link_trans_dic['object_name'].replace("-", "_")
        # 这是设置single side的基本条件
        omni.kit.commands.execute('ChangeSetting',path='/rtx/hydra/faceCulling/enabled',value=True)
        # get the dir {link_0: 'line_fixed', link_1: ......}
        link_type = {link: key for key, links in CAD_model_part_class_link_list.items() for link in links}

        # load all the object and set the single sided to true
        for link in CAD_model_link_name_mesh_list.keys():    # e.g. link: "link"
            for object in CAD_model_link_name_mesh_list[link]:   # e.g. object: "'/home/ubuntu/.local/share/ov/pkg/isaac_sim-2023.1.0-hotfix.1/isaac_works/cwb_part/gapartnet_cad_models/45146/textured_objs/original-43.obj'"
                prim_name = os.path.basename(object).split(".obj")[0] # e.g. prim_name:'original-43'
                # Note the difference between "_" and "-".
                part_prim = prims_utils.create_prim(
                    prim_path=f"/World/{prim_name}".replace("-", "_"),
                    # usd_path=f"/data/Part/GAPartNet_PartNetMobility/partnet_mobility_part/{CAD_MODEL}/usds/{prim_name}.usd",
                    usd_path=os.path.join(cad_model_dir, CAD_MODEL, "converted_usds", f"{prim_name}.usd"),
                    semantic_label=link_type[f"{link}"] + "_" + link if link != "base" else None,  # encode the part class and link name into the semantic label, sem_seg_map and ins_seg_map can be generated from this
                    orientation= torch.tensor([0.707,0.707,0.0,0.0])) # euler_angles_to_quat([math.radians(90), 0, 0]))
            
                # set single sided
                for mesh_Xform in part_prim.GetChildren():
                    if mesh_Xform.GetTypeName() == "Xform":
                        for mesh_prim in mesh_Xform.GetChildren():
                            # 设置single sided
                            bool_attr: Usd.Attribute = create_bool_attribute(mesh_prim, "singleSided")
                            bool_attr.Set(True)

        # create link and set the translate
        for key in CAD_model_link_name_mesh_list.keys(): # key: link_0
            if key=='base':
                continue
            # if need_change_material: 
            #     material_type = part_material_pairs[link_type[key]]
            #     mr = material_randomization(key, my_world, dr, material_type=material_type)
            link_prim = my_world.stage.GetPrimAtPath(f"/{link_trans_dic['object_name']}/{key}".replace("-", "_")) # link_prim: /xxx/link_0

            # delete the object in urdf
            for link_child_prim in link_prim.GetChildren():  
                prim_name = os.path.basename(link_child_prim.GetPath().pathString)  # prim_name: /xxx/visuals
                if prim_name in ["visuals", "Looks"]:
                    omni.kit.commands.execute('DeletePrims',
                        paths=[f"/{link_trans_dic['object_name']}/{key}/{prim_name}"],
                        destructive=False)

            # move the correct object into urdf structure so that we don't need to create joint
            part_path_list = CAD_model_link_name_mesh_list[key]

            for part_path in part_path_list:   # part_path: textured_objs/roiginal-114.obj
                name = os.path.basename(part_path).split(".obj")[0]
                omni.kit.commands.execute(
                        "MovePrim",
                        path_from= "/World/{}".format(name.replace("-", "_")),
                        path_to= "/{}/{}/{}".format(link_trans_dic['object_name'], key, name.replace("-", "_")),
                        keep_world_transform = True, destructive=False)  
                prim_path = my_world.stage.GetPrimAtPath('/' + link_trans_dic['object_name'] + '/' + key + '/' + name.replace("-", "_")) 

                prim_path.GetAttribute('xformOp:translate').Set((link_trans_dic[name][0][0],link_trans_dic[name][0][1],link_trans_dic[name][0][2]))
                prim_path.GetAttribute('xformOp:orient').Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

                # modify material, material_type=['specular', 'metal', 'diffuse', 'transparent', 'glass']
                # if need_change_material:
                #     mr.change_material(my_world, dr, "/{}/{}/{}/{}".format(link_trans_dic['object_name'], key, name.replace("-", "_"), name.replace("-", "_")), material_type)

        prim_path = my_world.stage.GetPrimAtPath(f"/{link_trans_dic['object_name']}/base") 
        prim_path.GetAttribute('xformOp:translate').Set((0, 0, height))



    def get_mobility(self):
        mobility = Mobility(self.default_zero_env_path + "/mobility", name="mobility")
        self._sim_config.apply_articulation_settings(
            "mobility", get_prim_at_path(mobility.prim_path), self._sim_config.parse_actor_config("mobility")
        )

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

    def get_observations(self) -> dict:
        hand_pos, hand_rot = self._frankas._hands.get_world_poses(clone=False)
        drawer_pos, drawer_rot = self._mobilitys._drawers.get_world_poses(clone=False)
        cabinet_pos,cabinet_rot = self._mobilitys.get_world_poses(clone=False)
        # print("cabinet_pos:",cabinet_pos)
        # print("cabinet_rot:",cabinet_rot)
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)
        self.mobility_dof_pos = self._mobilitys.get_joint_positions(clone=False)
        self.mobility_dof_vel = self._mobilitys.get_joint_velocities(clone=False)
        # print(self.mobility_dof_pos,"self.mobility_dof_pos")
        
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

        # reset mobility
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
        
        
        # 6. how far the mobility has been opened out    #weight=7.5
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
