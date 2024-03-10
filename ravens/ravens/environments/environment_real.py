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
                 task_name=None,
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
        print("Franka Client Start")
        self.task_name = task_name
        
        context_speech = zmq.Context()
        self.socket_speech = context_speech.socket(zmq.REQ)  # REQ (REQUEST) 소켓
        self.socket_speech.connect("tcp://115.145.178.235:5557")
        self.socket_speech.setsockopt(zmq.RCVTIMEO, 15000)

        self.audio_file_path = "/home/pjw971022/Sembot/real_bot/perception/speech_command.wav"
        print("Speech Client Start")
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
    def get_speech(self):
        speech_request_json = json.dumps('require speech')
        self.socket_speech.send_string(speech_request_json)
        print("speech client sent!!")
        try:
            audio_data = self.socket_speech.recv()
            with open(self.audio_file_path, "wb") as audio_file:
                audio_file.write(audio_data)
            print("speech client received!!")
        except zmq.Again as e:
            print("Waiting for a message timed out")

    def reset(self):
        if not isinstance(self.task, str):
            self.task.reset()
        obs, _, _, info = self.step(reset=True)
        return obs, info

    def step(self, raw_action=None, reset=False):
        self.step_counter += 1
        
        if raw_action is None:
            action = 0
        elif isinstance(raw_action, int):
            action = raw_action
        else:
            mode = raw_action['mode']
            pick_pose = raw_action['pose0']
            place_pose = raw_action['pose1']
            action = (mode, pick_pose, place_pose)

        obs = {'color': (), 'depth': (),'pointcloud': ()}
        for config in self.agent_cams:
            color, depth, pointcloud = self.get_data(config, action, reset=reset)
            obs['color'] += (color,)
            obs['depth'] += (depth,)
            obs['pointcloud'] += (pointcloud,)
        info = {}
        return obs, 0.0, False, info
    
    def get_obs_human_cam(self, mode):
        if mode == 1:
            for config in self.agent_cams:
                self.get_data(config, send_msg="require human-cam image")
        elif mode == 2:
            for config in self.agent_cams:
                self.get_data(config, send_msg="require human-cam video")
        else:
            raise ValueError("mode should be 1 or 2")

    def server_close(self):
        print("Send Server Close!")        
        go_to_ready_json = json.dumps(1)
        self.socket.send_string(go_to_ready_json)

    def get_data(self, config, send_msg, image_size=None, reset=False):
        if  isinstance(send_msg, str) and "human-cam" in send_msg:    
            msg_json = json.dumps(send_msg)
            self.socket.send_string(msg_json)
            print("human-cam client sent!!")
            if send_msg == "require human-cam image":
                image_data = self.socket.recv()
                with open('/home/pjw971022/Sembot/real_bot/save_vision/obs/human_image.png', 'wb') as image_file:
                    image_file.write(image_data)
            elif send_msg == "require human-cam video":
                video_data = self.socket.recv()
                with open('/home/pjw971022/Sembot/real_bot/save_vision/obs/human_video.mp4', 'wb') as video_file:
                    video_file.write(video_data)
            return None
        else:
            
            msg_json = json.dumps(send_msg)
            self.socket.send_string(msg_json)
            print("franka client sent!!")

            data = self.socket.recv_string()
            print("franka client received!!")
            data = json.loads(data)
            color = np.array(data['rgb'])
            depth = np.array(data['depth'])
            
            cv2.imwrite('/home/pjw971022/Sembot/real_bot/save_vision/obs/image_obs.png', color)

            """Render RGB-D image with specified camera configuration."""
            if not image_size:
                image_size = config['image_size']

            if 'pointcloud' in data.keys():
                pointcloud = np.array(data['pointcloud']).reshape(image_size[0], image_size[1],-1)

            znear, zfar = config['zrange']

            # Get depth image.
            # depth_image_size = (image_size[0], image_size[1])
            depth = (zfar + znear - (2. * depth - 1.) * (zfar - znear))
            depth = (2. * znear * zfar) / depth
            # if config['noise']:
            #     depth += self._random.normal(0, 0.003, depth_image_size)

            return color, depth, pointcloud

    @property
    def info(self):
        """Environment info variable with object poses, dimensions, and colors."""

        info = {}  # object id : (position, rotation, dimensions)
        info['lang_goal'] = self.get_lang_goal()
        info['final_goal'] = ''
        return info

    def set_task(self, task):
        self.task = task
        if not isinstance(task, str):
            self.objects = task.objects
            self.categories = task.categories
            self.receptacles = task.receptacles

    def get_lang_goal(self):
        return self.task.get_lang_goal()
    


    # def get_final_goal(self):
    #     if self.task:
    #         return self.task.get_final_lang_goal()
    #     else:
    #         raise Exception("No task for was set")
    


