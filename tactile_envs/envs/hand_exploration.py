import os
import cv2 

import gymnasium as gym
import time
import mujoco

import numpy as np
from gymnasium import spaces
import cv2

from pathlib import Path

def convert_observation_to_space(observation, compress_img: bool = False):
    
    space = spaces.Dict(spaces={})
    for key in observation.keys():
        if key == 'image':
            if compress_img:
                space.spaces[key] = spaces.Box(low = 0, high = 255, shape = observation[key].shape, dtype = np.uint8)
            else:
                space.spaces[key] = spaces.Box(low = 0, high = 1, shape = observation[key].shape, dtype = np.float64)
        elif key == 'state':
            space.spaces[key] = spaces.Box(low = -float('inf'), high = float('inf'), shape = observation[key].shape, dtype = np.float64)
        
    return space


class HandExplorationEnv(gym.Env):

    def __init__(self, start_grasped = True, state_type='vision', camera_idx=0,
        env_id = -1, im_size=64, skip_frame=10, max_delta=None, multiccd=False,
        compress_img: bool = True,
        num_init_grasp_steps: int = 0,
        return_grasp_flag_as_reward: bool = False,
        initialize_assets: bool = False,
        multi_obj: bool = False,
        ):

        """
        'no_rotation': if True, the robot will not be able to rotate its wrist
        'no_gripping': if True, the robot will keep the gripper opening at a fixed value
        'state_type': choose from 'privileged', 'vision', 'touch', 'vision_and_touch'
        'camera_idx': index of the camera to use
        'env_id': environment id
        'im_size': side of the square image
        'skip_frame': number of frames to skip between actions
        'max_delta': maximum change allowed in the x, y, z position
        'multiccd': if True, the multiccd flag will be enabled (makes tactile sensing more accurate but slower)
        'objects': list of objects to insert (list from "square", "triangle", "horizontal", "vertical", "trapezoidal", "rhombus")
        'holders': list of holders to insert the objects (list from "holder1", "holder2", "holder3")
        """

        super(HandExplorationEnv, self).__init__()

        self.id = env_id

        self.compress_img = compress_img

        self.skip_frame = skip_frame
        
        asset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets')

        original_dir = os.getcwd()

        # Change the working directory to 'tactile_envs'
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
        
        # Change the working directory back to the original one
        os.chdir(original_dir)

        assert multi_obj == False, "multi_obj not supported yet"
        assert start_grasped == False, "start_grasped not supported yet"

        if multi_obj:
            self.model_path = os.path.join(asset_folder, 'shadow_hand_submodel/scene_multi.xml')
        else:
            self.model_path = os.path.join(asset_folder, 'shadow_hand_submodel/scene_single.xml')
        self.multi_obj = multi_obj
        self.current_dir = os.path.join(Path(__file__).parent.parent.absolute(), 'assets/shadow_hand_submodel')
        with open(self.model_path,"r") as f:
            self.xml_content = f.read()
        self.update_include_path()
        self.xml_content_reference = self.xml_content

        self.multiccd = multiccd

        self.fixed_gripping = 200

        self.max_delta = max_delta

        self.im_size = im_size

        self.state_type = state_type

        print("state_type: ", self.state_type)

        if self.state_type == 'privileged':
            self.curr_obs = {'state': np.zeros(40)}
        elif self.state_type == 'vision':
            if self.compress_img:
                self.curr_obs = {'image': np.zeros((self.im_size, self.im_size, 3), dtype=np.uint8)}
            else:
                self.curr_obs = {'image': np.zeros((self.im_size, self.im_size, 3))}
        else:
            raise ValueError("Invalid state type")
        
        self.sim = self.from_xml_string()
        self.mj_data = mujoco.MjData(self.sim)
        if self.multiccd:
            self.sim.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD

        self.init_z = self.mj_data.qpos[-5]

        self.start_grasped = start_grasped
        self.num_init_grasp_steps = num_init_grasp_steps
        self.num_env_steps = 0

        self.camera_idx = camera_idx        
        
        obs_tmp = self._get_obs()
        self.observation_space = convert_observation_to_space(obs_tmp, compress_img)
        
        self.ndof_u = self.sim.nu

        print("ndof_u: ", self.ndof_u)
        
        self.action_space = spaces.Box(low=np.full(self.ndof_u, -1., dtype=np.float32),
                                       high=np.full(self.ndof_u, 1., dtype=np.float32),
                                       dtype = np.float32)
        # self.action_scale = np.array([[-0.2,0.2],[-0.2,0.2],[-0.12,0.3],[-np.pi,np.pi],[0,220]])
        self.action_scale = self.sim.actuator_ctrlrange.copy()
        print("action_scale: ", self.action_scale)

        # self.action_mask = np.ones(5, dtype=bool)
        
        # self.action_scale = self.action_scale[self.action_mask]
        
        self.renderer = mujoco.Renderer(self.sim, height=self.im_size, width=self.im_size)
        self.return_grasp_flag_as_reward = return_grasp_flag_as_reward

    def from_xml_string(self):
        timeout = 120
        start_time = time.time()
        while True:
            try:
                sim = mujoco.MjModel.from_xml_string(self.xml_content)
                break
            except Exception as e:
                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout:
                    raise AssertionError(f"Failed to initialize simulation within {timeout} seconds: {e}")
                time.sleep(1)  # Wait for 1 second before retrying
                print(elapsed_time)
        return sim

    def update_include_path(self):
        
        file_idx = self.xml_content.find('<include file="', 0)
        while file_idx != -1:
            file_start_idx = file_idx + len('<include file="')
            self.xml_content = self.xml_content[:file_start_idx] + self.current_dir + '/' + self.xml_content[file_start_idx:]

            file_idx = self.xml_content.find('<include file="', file_start_idx + len(self.current_dir))

        file_idx = self.xml_content.find('meshdir="', 0)
        file_start_idx = file_idx + len('meshdir="')
        self.xml_content = self.xml_content[:file_start_idx] + self.current_dir + '/' + self.xml_content[file_start_idx:]

    @property
    def grasp_object(self):
        if self.start_grasped:
            return True
        else:
            if self.num_env_steps >= self.num_init_grasp_steps:
                return False
            else:
                return True

    def generate_initial_pose(self, show_full=False):
        
        print("resetting initial pose")
        cruise_height = 0.
        
        mujoco.mj_resetData(self.sim, self.mj_data)

        rand_x = np.random.rand()*0.4 - 0.2
        rand_y = np.random.rand()*0.4 - 0.2
        if self.with_rotation:
            rand_yaw = np.random.rand()*2*np.pi - np.pi
        else:
            rand_yaw = 0

        steps_per_phase = 60

        placed_coords = np.empty((0,2))

        if self.grasp_object:
            mujoco.mj_resetDataKeyframe(self.sim, self.mj_data, 0)

            for i in range(steps_per_phase): # rotate in place
                self.mj_data.ctrl[3] = -rand_yaw
                mujoco.mj_step(self.sim, self.mj_data, self.skip_frame+1)
                if show_full:
                    self.renderer.update_scene(self.mj_data, camera=0)
                    img = cv2.cvtColor(self.renderer.render(), cv2.COLOR_BGR2RGB)
                    cv2.imshow('img', img)
                    cv2.waitKey(1)
                
            for i in range(steps_per_phase): # move to random position
                self.mj_data.ctrl[:3] = [rand_x, rand_y, cruise_height]
                mujoco.mj_step(self.sim, self.mj_data, self.skip_frame+1)
                if show_full:
                    self.renderer.update_scene(self.mj_data, camera=0)
                    img = cv2.cvtColor(self.renderer.render(), cv2.COLOR_BGR2RGB)
                    cv2.imshow('img', img)
                    cv2.waitKey(1)
        else:
            self.mj_data.joint('object1_jnt').qpos[0:2] = [rand_x, rand_y]
            self.mj_data.joint('object1_jnt').qpos[3:7] = [np.cos(rand_yaw/2), 0, 0, np.sin(rand_yaw/2)]
            placed_coords = np.vstack((placed_coords, np.array([rand_x, rand_y])))

        if self.multi_obj:
            for i in range(2, 8):
                dist = 0
                attempt = 0
                while dist < 0.075:
                    rand_x = np.random.rand()*0.4 - 0.2
                    rand_y = np.random.rand()*0.4 - 0.2
                    if self.with_rotation:
                        rand_yaw = np.random.rand()*2*np.pi - np.pi
                    else:
                        rand_yaw = 0
                    point = np.array([rand_x, rand_y])
                    if len(placed_coords) == 0:
                        break
                    dist = np.min(np.linalg.norm(placed_coords - point, axis=1))
                    attempt += 1
                    if attempt > 20:
                        print("Too many attempts to place object")
                        break
                self.mj_data.joint(f'object{i}_jnt').qpos[0:2] = point
                self.mj_data.joint(f'object{i}_jnt').qpos[3:7] = [np.cos(rand_yaw/2), 0, 0, np.sin(rand_yaw/2)]
                placed_coords = np.vstack((placed_coords, point))
        
        self.prev_action_xyz = self.mj_data.ctrl[:3].copy()

        mujoco.mj_forward(self.sim, self.mj_data)

    def _get_obs(self):
        return self.curr_obs
    
    def get_proprio(self):
        left_finger = self.mj_data.site("finger_left").xpos
        right_finger = self.mj_data.site("finger_right").xpos
        distance = np.linalg.norm(left_finger - right_finger)
        robot_state = np.concatenate([self.mj_data.qpos.copy()[:4], [distance]])
        return robot_state
    
    def seed(self, seed):
        np.random.seed(seed)
    
    def reset(self, seed=None, options=None):

        print("resetting environment")

        if seed is not None:
            np.random.seed(seed)
        
        self.sim = self.from_xml_string()
        
        self.mj_data = mujoco.MjData(self.sim)
        if self.multiccd:
            self.sim.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD 
        
        del self.renderer
        self.renderer = mujoco.Renderer(self.sim, height=self.im_size, width=self.im_size)

        # self.generate_initial_pose()

        if self.state_type == 'vision':
            img = self.render()
            self.curr_obs = {'image': img}
        elif self.state_type == 'privileged':
            self.curr_obs = {'state': np.concatenate((self.mj_data.qpos.copy(), self.mj_data.qvel.copy(), [self.offset_x,self.offset_y,self.offset_yaw]))}
        
        info = {'id': np.array([self.id]),
                'is_success': int(False),
                'grasped': int(self.verify_grasp()),
                }

        mujoco.mj_forward(self.sim, self.mj_data)

        return self._get_obs(), info


    def render(self, highres = False):
        
        if highres:
            del self.renderer
            self.renderer = mujoco.Renderer(self.sim, height=480, width=480)
            self.renderer.update_scene(self.mj_data, camera=self.camera_idx)
            img = self.renderer.render() # /255
            del self.renderer
            self.renderer = mujoco.Renderer(self.sim, height=self.im_size, width=self.im_size)
        else:
            self.renderer.update_scene(self.mj_data, camera=self.camera_idx)
            img = self.renderer.render() #/255
        if self.compress_img:
            return img.astype(np.uint8)
        else:
            return img / 255

    def step(self, u):
        self.num_env_steps += 1
        grasped = self.verify_grasp()

        start_idx_act = 20

        action = u
        action = np.clip(u, -1., 1.)
        
        action_unnorm = (action + 1)/2 * (self.action_scale[:,1]-self.action_scale[:,0]) + self.action_scale[:,0]

        if self.max_delta is not None:
            action_unnorm[start_idx_act:start_idx_act+3] = np.clip(action_unnorm[start_idx_act:start_idx_act+3], self.prev_action_xyz - self.max_delta, self.prev_action_xyz + self.max_delta)
        
        self.prev_action_xyz = action_unnorm

        self.mj_data.ctrl = action_unnorm
        self.mj_data.ctrl[start_idx_act:start_idx_act+3] = action_unnorm[start_idx_act:start_idx_act+3]

        mujoco.mj_step(self.sim, self.mj_data, self.skip_frame+1)
        
        if self.state_type == 'vision':
            img = self.render()
            self.curr_obs = {'image': img}
            info = {'id': np.array([self.id])}
        elif self.state_type == 'privileged':
            self.curr_obs = {'state': np.concatenate((self.mj_data.qpos.copy(), self.mj_data.qvel.copy(), [self.offset_x,self.offset_y,self.offset_yaw]))}
            info = {'id': np.array([self.id])}

        done = False
        info['is_success'] = int(done)
        info['grasped'] = int(grasped)

        obs = self._get_obs()

        reward = float(grasped) if self.return_grasp_flag_as_reward else 0.0

        return obs, reward, done, False, info

    def verify_grasp(self):
        grasped = False
        for pair in self.mj_data.contact.geom:
            pair_0 = self.sim.geom(pair[0]).name
            pair_1 = self.sim.geom(pair[1]).name
            if (pair_0.startswith("holder_collision") or pair_0.startswith("peg_collision")) and (pair_1==""):
                grasped = True
                break
            elif (pair_1.startswith("holder_collision") or pair_1.startswith("peg_collision")) and (pair_0==""):
                grasped = True
                break
        return grasped
        