from video_utils import Video2TextDataGenrator
import pandas as pd
import numpy as np
import glob

from rlbench import tasks
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import ArmActionMode, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete, GripperActionMode
from rlbench.observation_config import ObservationConfig, CameraConfig
# from sembot.src.utils import normalize_vector, bcolors

class CustomMoveArmThenGripper(MoveArmThenGripper):
    """
    A potential workaround for the default MoveArmThenGripper as we frequently run into zero division errors and failed path.
    TODO: check the root cause of it.
    Ignore arm action if it fails.

    Attributes:
        _prev_arm_action (numpy.ndarray): Stores the previous arm action.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prev_arm_action = None

    def action(self, scene, action):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        # if the arm action is the same as the previous action, skip it
        if self._prev_arm_action is not None and np.allclose(arm_action, self._prev_arm_action):
            self.gripper_action_mode.action(scene, ee_action)
        else:
            try:
                self.arm_action_mode.action(scene, arm_action)
            except Exception as e:
                print(f'[rlbench_env.py] Ignoring failed arm action; Exception: "{str(e)}"') # @
            self.gripper_action_mode.action(scene, ee_action)
        self._prev_arm_action = arm_action.copy()

from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy

from rlbench.backend.observation import Observation
from rlbench.backend.task import Task
from rlbench.sim2real.domain_randomization import RandomizeEvery, \
    VisualRandomizationConfig

from typing import Type
# from absl import app
# from absl import flags
import os


class CameraMotion(object):
    def __init__(self, cam: VisionSensor):
        self.cam = cam

    def step(self):
        raise NotImplementedError()

    def save_pose(self):
        self._prev_pose = self.cam.get_pose()

    def restore_pose(self):
        self.cam.set_pose(self._prev_pose)


class CircleCameraMotion(CameraMotion):

    def __init__(self, cam: VisionSensor, origin: Dummy, speed: float):
        super().__init__(cam)
        self.origin = origin
        self.speed = speed  # in radians

    def step(self):
        self.origin.rotate([0, 0, self.speed])

import json
from PIL import Image
from rlbench.backend import utils

class TaskRecorder(object):
    def __init__(self, env: Environment,
                 cam_motion: CameraMotion,
                 fps=30, prompt='motion',
                 save_dir='/tmp/data/videos',
                 gen_text=True,
                 variations=-1):
        self._env = env
        self._cam_motion = cam_motion
        self._fps = fps
        self._snaps = []
        self._current_snaps = []

        self.v2t_generator = Video2TextDataGenrator()
        self.df = pd.DataFrame()
        self.prompt = prompt
        self.save_dir = save_dir
        self.gen_text = gen_text

        self.variations = variations
        fewshot_path = '/home/jinwoo/workspace/Sembot/physical_reasoning/fewshot.txt'
        with open(fewshot_path, 'r') as f:
            self.fewshot = f.read().strip()

        object_name_path = '/home/jinwoo/workspace/Sembot/sembot/src/envs/task_object_names.json'
        with open(object_name_path, 'r') as f:
            self.task_object_names = json.load(f)

    def get_object_names(self, task_name: str):
        """
        Returns the names of all objects in the current task environment.

        Returns:
            list: A list of object names.
        """
        name_mapping = self.task_object_names[task_name]
        exposed_names = [names[0] for names in name_mapping]
        return exposed_names
    
    def take_snap(self, obs: Observation,reset=False):
        if reset:
            self._cam_motion.restore_pose()
            self._current_snaps = []
        else:
            self._cam_motion.step()
            self._current_snaps.append(
                (self._cam_motion.cam.capture_rgb() * 255.).astype(np.uint8))
        
    def observation_save(self, task_name, obs, ep_idx=0):
        left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb)
        right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
        overhead_rgb = Image.fromarray(obs.overhead_rgb)
        wrist_rgb = Image.fromarray(obs.wrist_rgb)
        front_rgb = Image.fromarray(obs.front_rgb)

        obs_save_dir = self.save_dir + f'/{task_name}/obs/'
        os.makedirs(obs_save_dir, exist_ok=True)

        left_shoulder_rgb.save(
            os.path.join(obs_save_dir, f'left_shoulder_rgb__ep{ep_idx}.png'))
        right_shoulder_rgb.save(
            os.path.join(obs_save_dir, f'right_shoulder_rgb__ep{ep_idx}.png'))
        overhead_rgb.save(
            os.path.join(obs_save_dir, f'overhead_rgb__ep{ep_idx}.png'))
        wrist_rgb.save(
            os.path.join(obs_save_dir, f'wrist_rgb__ep{ep_idx}.png'))
        front_rgb.save(
            os.path.join(obs_save_dir, f'front_rgb__ep{ep_idx}.png'))
        return obs_save_dir
    
    def record_task(self, task: Type[Task], vr_cnt=0, ep_idx=0):
        task_env = self._env.get_task(task)
        task_name = task_env.get_name()
        save_dir = self.save_dir + f'/{task_name}/'
        os.makedirs(save_dir, exist_ok=True)

        var_target = task_env.variation_count()
        if self.variations >= 0:
            var_target = np.minimum(self.variations, var_target)
        if vr_cnt >= var_target:
            return False
        task_env.set_variation(vr_cnt)

        self._cam_motion.save_pose()
        max_try = 5
        try_cnt=0
        while max_try>try_cnt:
            try_cnt+=1
            try:
                video_file_path = save_dir+f'ep{ep_idx}_video.mp4'
                if os.path.exists(video_file_path):
                    print(f"Skipping {task_name} episode {ep_idx} as it already exists.")
                    desc_ = task_env._scene.init_episode(0)
                else:
                    demos, desc_list, obs_list = task_env.get_demos(
                        1, live_demos=True, callable_each_step=self.take_snap,
                        max_attempts=5)
                    self._snaps.extend(self._current_snaps)
                    self._current_snaps = []

                    self.save(save_dir,ep_idx=ep_idx)
                    desc_ = desc_list[0]
                    obs_ = obs_list[0] 

                if self.gen_text:
                    objects = self.get_object_names(task_name)
                    if len(desc_)==2:
                        descriptions, target_obj = desc_
                    else:
                        descriptions = desc_    
                    instruction = np.random.choice(descriptions) #f"move the {target_obj} to the cupboard." 
                    if self.prompt == 'naive':
                        prompt_ = "please describe the video."
                    elif self.prompt == 'key_feature':
                        prompt_ = "please describe the key features of the video."
                    elif self.prompt == 'motion':
                        system_prompt_ = 'I want to gather information from a video. Please represent the actions in the video as a plan, using only the predefined skills: actions, move, grasp, and rotate.'
                        prompt_ = f"I will give you some example of plan. {self.fewshot}\nobjects = {objects}\n#Query: {instruction}."

                    description = self.v2t_generator.v2t_generate(video_file_path, system_prompt_, prompt_)
                    self.df = self.df.append({"task_name":task_name, "episode_idx":ep_idx, "description": description}, ignore_index=True)
                break
            except RuntimeError:
                self._cam_motion.restore_pose()
                self._current_snaps = []

        return True

    def save(self, path, ep_idx, save_init_obs=True):
        print('Converting to video ...')
        video_file_path = os.path.join(path, f'ep{ep_idx}_video.mp4')

        # OpenCV QT version can conflict with PyRep, so import here
        import cv2
        video = cv2.VideoWriter(
                video_file_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), self._fps,
                tuple(self._cam_motion.cam.get_resolution()))
        for image in self._snaps:
            video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        video.release()

        # Save first image separately if requested
        if save_init_obs and self._snaps:
            init_obs_path = os.path.join(path, f'ep{ep_idx}_init_obs.jpg') 
            init_obs = self._snaps[0]
            cv2.imwrite(init_obs_path, cv2.cvtColor(init_obs, cv2.COLOR_RGB2BGR))
        self._snaps = []

from rlbench.backend.utils import task_file_to_task_class
from rlbench.backend.task import TASKS_PATH
def main(args):
    action_mode = CustomMoveArmThenGripper(arm_action_mode=EndEffectorPoseViaPlanning(),
                                    gripper_action_mode=Discrete())
    # cam_config = CameraConfig(image_size=(720,720))
    # obs_config = ObservationConfig( right_shoulder_camera= cam_config,
    #                                 left_shoulder_camera= cam_config,
    #                                 overhead_camera= cam_config,
    #                                 wrist_camera= cam_config,
    #                                 front_camera= cam_config,
    #                                 record_gripper_closing=True)
    obs_config = ObservationConfig(record_gripper_closing=True)
    obs_config.set_all(False)

    print("Headless mode: ", args.headless)
    env = Environment(action_mode, obs_config=obs_config,
                      headless=args.headless)
    env.launch()

    # Add the camera to the scene
    cam_placeholder = Dummy('cam_cinematic_placeholder')
    cam = VisionSensor.create(args.camera_resolution)
    cam.set_pose(cam_placeholder.get_pose())
    cam.set_parent(cam_placeholder)

    cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), 0.000)
    tr = TaskRecorder(env,
                      cam_motion,
                      fps=30,
                      prompt=args.prompt,
                      save_dir=args.save_dir,
                      gen_text= args.gen_text,
                      variations=args.variations,)

    #task_list = [
    #    tasks.PourFromCupToCup
    #]
    task_names = [t.split('.py')[0] for t in os.listdir(TASKS_PATH)
                      if t != '__init__.py' and t.endswith('.py')]
    task_classes = [task_file_to_task_class(
        task_file) for task_file in task_names]
    task_list = list(zip(task_names, task_classes))


    for _, task in task_list:
        for i in range(args.episodes_per_task):
            print("Start recording task: ", task.__name__, f"episode: {i}")
            good = tr.record_task(task, vr_cnt=0, ep_idx=i)
    tr.df.to_csv("video2text.csv", index=False)
    env.shutdown()

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RLBench Task Recorder Settings')

    parser.add_argument(
        '--save_dir', default='/home/jinwoo/data/rlbench/',
        help='Where to save the generated videos.')
    parser.add_argument(
        '--gen_text', action='store_true',
        help='One long clip of all the tasks, or individual videos.')
    parser.add_argument(
        '--individual', action='store_true',
        help='One long clip of all the tasks, or individual videos.')
    parser.add_argument(
        '--prompt', default='motion',
        help='Prompt mode.')
    parser.add_argument(
        '--headless', action='store_true',
        help='Run in headless mode.')
    parser.add_argument(
        '--camera_resolution', nargs=2, default=[1280, 720], type=int,
        help='The camera resolution')
    parser.add_argument(
        '--variations', type=int, default=1,
        help='Number of variations to collect per task. -1 for all.')
    parser.add_argument(
        '--episodes_per_task', type=int, default=1,
        help='The number of episodes to collect per task.')

    args = parser.parse_args()
    main(args)
