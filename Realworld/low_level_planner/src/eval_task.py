import hydra
from omegaconf import DictConfig, OmegaConf
import openai
from arguments import get_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from envs.rlbench_env import VoxPoserRLBench
from utils import set_lmp_objects
import numpy as np
from rlbench import tasks
import os
from envs.motion_descriptor import MotionDescriptor
import envs.prompts as prompts
# os.environ['OPENAI_API_KEY'] =  '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' 
import wandb
from PIL import Image
@hydra.main(config_path=f'./configs', config_name="rlbench_config")
def main(cfgs: DictConfig):
    config = cfgs
    # uncomment this if you'd like to change the language model (e.g., for faster speed or lower cost)
    for lmp_name, cfg in config['lmp_config']['lmps'].items():
        cfg['model'] = 'gemini-pro'

    # initialize env and voxposer ui
    visualizer = ValueMapVisualizer(config['visualizer'])
    env = VoxPoserRLBench(visualizer=visualizer)
    lmps, lmp_env = setup_LMP(env, config, debug=False)
    voxposer_ui = lmps['plan_ui']
    task_list = [
    # tasks.PutRubbishInBin,
    #  tasks.LampOff,
    #  tasks.OpenWineBottle,
     tasks.PushButton,
     tasks.TakeOffWeighingScales,
     tasks.MeatOffGrill,
     tasks.SlideBlockToTarget,
     tasks.TakeLidOffSaucepan,
     tasks.TakeUmbrellaOutOfUmbrellaStand,
     tasks.PutGroceriesInCupboard
     ]
    # below are the tasks that have object names added to the "task_object_names.json" file
    # uncomment one to use
    plain_dict = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    # wandb.init(project="voxposer-naive", entity="shyuni5/file/CORL2024", name=config.context_mode, config={})
    for task in task_list:
        env.load_task(task)
        task_name = env.task.get_name()
        wandb.init(project="voxposer-naive",
                   entity="shyuni5/file/CORL2024",
                   name=task_name, 
                   reinit=True)
        wandb.config.update(plain_dict)

        for i in range(cfgs['episode_length']):
            descriptions, obs = env.reset()
            set_lmp_objects(lmps, env.get_object_names())  # set the object names to be used by voxposer
            instruction = np.random.choice(descriptions)
            print(f'instruction: {instruction}')
            if 'pre' in config.context_mode:
                if env.task.get_name() in prompts.names.keys():
                    prompt_cls = prompts.names[env.task.get_name()]()
                    if 'user_command' in config.context_mode:
                        motion_guideline = prompt_cls.get_u2c()
                    if 'vision_observation' in config.context_mode:
                        motion_guideline = prompt_cls.get_o2c()
                    if 'expert_demo' in config.context_mode:
                        motion_guideline = prompt_cls.get_d2c()
                else:
                    motion_guideline = None
            elif 'gen' in config.context_mode:
                frong_rgb = Image.open(f'/home/jinwoo/workspace/Sembot/low_level_planner/src/visualizations/obs/{task_name}/front_rgb.png')
                expert_demo_video = None
                motion_descriptor = MotionDescriptor()
                if 'user_command' in config.context_mode:
                    motion_guideline = motion_descriptor.gemini_gen_u2c(user_command=instruction)
                elif 'vision_observation' in config.context_mode:
                    motion_guideline = motion_descriptor.gemini_gen_o2c(user_command=instruction,
                                                                        img=frong_rgb)
                elif 'expert_demo' in config.context_mode:
                    motion_guideline = motion_descriptor.gemini_gen_d2c(user_command=instruction,
                                                                        img=frong_rgb,
                                                                        video=expert_demo_video)
            else:
                motion_guideline = None
            
            voxposer_ui(instruction,
                        motion_guideline=motion_guideline)
            # Log task name to wandb
            
    wandb.finish()
#     for task in task_list:
#         env.load_task(task)
#         descriptions, obs = env.reset()
#         set_lmp_objects(lmps, env.get_object_names())  # set the object names to be used by voxposer
#         task_name = env.task.get_name()
#         instruction = np.random.choice(descriptions)
#         print(f'instruction: {instruction}')

#         if 'pre' in  config.context_mode:
#             if env.task.get_name() in prompts.names.keys():
#                 prompt_cls = prompts.names[env.task.get_name()]()
#                 if 'user_command' in  config.context_mode:
#                     motion_guideline = prompt_cls.get_u2c()
#                 if 'vision_observation' in  config.context_mode:
#                     motion_guideline = prompt_cls.get_o2c()
#                 if 'expert_demo' in  config.context_mode:
#                     motion_guideline = prompt_cls.get_d2c()
#             else:
#                 motion_guideline = None
#         elif 'gen' in  config.context_mode:
#             from PIL import Image
# import wandb
# from PIL import Image
#             frong_rgb = Image.open(f'/home/jinwoo/workspace/Sembot/low_level_planner/src/visualizations/obs/{task_name}/front_rgb.png')
#             expert_demo_video = None
#             motion_descriptor = MotionDescriptor()
#             if 'user_command' in  config.context_mode:
#                 motion_guideline = motion_descriptor.gemini_gen_u2c(user_command=instruction)
#             elif 'vision_observation' in  config.context_mode:
#                 motion_guideline = motion_descriptor.gemini_gen_o2c(user_command=instruction,
#                                                                     img=frong_rgb)
#             elif 'expert_demo' in  config.context_mode:
#                 motion_guideline = motion_descriptor.gemini_gen_d2c(user_command=instruction,
#                                                                     img=frong_rgb,
#                                                                     video=expert_demo_video)
#         else:
#             motion_guideline = None
#         voxposer_ui(instruction,
#                     motion_guideline=motion_guideline)

if __name__ =='__main__':
    main()