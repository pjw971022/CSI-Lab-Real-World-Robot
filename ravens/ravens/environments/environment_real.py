"""Environment class."""

import socket
import json

import gym
import numpy as np
from ravens.tasks import cameras
from ravens.utils import utils
import numpy as np

import zmq
import json
import cv2

# PLACE_STEP = 0.0003
# PLACE_DELTA_THRESHOLD = 0.005


class RealEnvironment(gym.Env):
    """OpenAI Gym-style environment class."""

    def __init__(self,
                 task=None,
                 ):
        """Creates OpenAI Gym-style environment with PyBullet.

        Args:
          assets_root: root directory of assets.
          task: the task to use. If None, the user must call set_task for the
            environment to work properly.
          disp: show environment with PyBullet's built-in display viewer.
          shared_memory: run with shared memory.
          hz: PyBullet physics simulation step speed. Set to 480 for deformables.

        Raises:
          RuntimeError: if pybullet cannot load fileIOPlugin.
        """
        self.pix_size = 0.003125
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
        self.homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.agent_cams = cameras.RealSenseD435.CONFIG
        # self.record_cfg = record_cfg
        self.save_video = False
        self.step_counter = 0
        self.pointcloud = None
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)  # REQ (REQUEST) 소켓
        self.socket.connect("tcp://115.145.175.206:5555")
        print("ROS Client Start")
        color_tuple = [
            gym.spaces.Box(0, 255, config['image_size'] + (3,), dtype=np.uint8)
            for config in self.agent_cams
        ]
        depth_tuple = [
            gym.spaces.Box(0.0, 20.0, config['image_size'], dtype=np.float32)
            for config in self.agent_cams
        ]
        self.observation_space = gym.spaces.Dict({
            'color': gym.spaces.Tuple(color_tuple),
            'depth': gym.spaces.Tuple(depth_tuple),
        })
        self.position_bounds = gym.spaces.Box(
            low=np.array([0.25, -0.5, 0.], dtype=np.float32),
            high=np.array([0.75, 0.5, 0.28], dtype=np.float32),
            shape=(3,),
            dtype=np.float32)
        self.action_space = gym.spaces.Dict({
            'pose0':
                gym.spaces.Tuple(
                    (self.position_bounds,
                     gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32))),
            'pose1':
                gym.spaces.Tuple(
                    (self.position_bounds,
                     gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)))
        })
        self.objects = None
        self.categories = None
        
    # ---------------------------------------------------------------------------
    # Standard Gym Functions
    # ---------------------------------------------------------------------------

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def reset(self):
        if not isinstance(self.task, str):
            self.task.reset()
        obs, _, _, _ = self.step()
        return obs
    
    def se3_to_pose(self, se3):
        p0_xyz, p0_xyzw = se3
        z = 0.1 # predict or heuristic
        yaw = 0 # predict or heuristic
        return (p0_xyz[0]-0.25, p0_xyz[1], z, 0, np.pi, yaw) # pose align constant
    
    def step(self, raw_action=None, cliport=False):
        self.step_counter += 1
        timeout = self.timeout()
        if timeout:
            obs = {'color': (), 'depth': (),'pointcloud': ()}
            for config in self.agent_cams:
                color, depth, pointcloud = self.render_camera(config)
                obs['color'] += (color,)
                obs['depth'] += (depth,)
                obs['pointcloud'] += (pointcloud,)
            return obs, 0.0, True, self.info
        
        if raw_action is None:
            action = 0
        elif isinstance(raw_action, int):
            action = raw_action
        else:
            if not cliport:
                pick_pose = raw_action['pose0']
                place_pose = raw_action['pose1']
                action = (pick_pose, place_pose)
            else:
                pick_pose = self.se3_to_pose(raw_action['pose0'])
                place_pose = self.se3_to_pose(raw_action['pose1'])
                action = (pick_pose, place_pose)

        action_json = json.dumps(action)
        self.socket.send_string(action_json)
        print("client sent !!")
        obs = {'color': (), 'depth': (),'pointcloud': ()}
        for config in self.agent_cams:
            color, depth, pointcloud = self.render_camera(config)
        
            obs['color'] += (color,)
            obs['depth'] += (depth,)
            obs['pointcloud'] += (pointcloud,)
        # Get task rewards.
        info = {}
        reward = self.reward()
        done = self.done(self.step_counter)
        return obs, reward, done, info
    
    def server_close(self):
        print("Send Server Close!")        
        go_to_ready_json = json.dumps(1)
        self.socket.send_string(go_to_ready_json)

    def timeout(self,):
        timeout = False
        return timeout
    
    def done(self, step_cnt):
        # if step_cnt > 1:
        #     done = True
        # else:
        done = False
        return done
    
    def reward(self,):
        reward = 0.
        return reward

    def render_camera(self, config, image_size=None): # render_camera
        data = self.socket.recv_string()
        print("client received !!")
        data = json.loads(data)

        color = np.array(data['rgb'])
        depth = np.array(data['depth'])
        
        # color = cv2.flip(color, -1)
        cv2.imwrite('/home/pjw971022/RealWorldLLM/save_viz/obs/image_obs.png', color)

        """Render RGB-D image with specified camera configuration."""
        if not image_size:
            image_size = config['image_size']

        if 'pointcloud' in data.keys():
            pointcloud = np.array(data['pointcloud']).reshape(image_size[0], image_size[1],-1)
        znear, zfar = config['zrange']
        
        # if config['noise']:
        #     color = np.int32(color)
        #     color += np.int32(self._random.normal(0, 3, image_size))
        #     color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (image_size[0], image_size[1])
        depth = (zfar + znear - (2. * depth - 1.) * (zfar - znear))
        depth = (2. * znear * zfar) / depth
        # if config['noise']:
        #     depth += self._random.normal(0, 0.003, depth_image_size)

        return color, depth, pointcloud

    @property
    def info(self):
        """Environment info variable with object poses, dimensions, and colors."""

        # Some tasks create and remove zones, so ignore those IDs.
        # removed_ids = []
        # if (isinstance(self.task, tasks.names['cloth-flat-notarget']) or
        #         isinstance(self.task, tasks.names['bag-alone-open'])):
        #   removed_ids.append(self.task.zone_id)

        info = {}  # object id : (position, rotation, dimensions)
        info['lang_goal'] = self.get_lang_goal()
        info['final_goal'] = self.get_final_goal()
        
        return info

    def set_task(self, task):
        self.task = task
        if not isinstance(task, str):
            self.objects = task.objects
            self.categories = task.categories
            self.receptacles = task.receptacles

    def get_lang_goal(self):
        return self.task.get_lang_goal()

    def get_final_goal(self):
        if self.task:
            return self.task.get_final_lang_goal()
        else:
            raise Exception("No task for was set")
    


