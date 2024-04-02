import hydra
from omegaconf import DictConfig, OmegaConf
import openai
from arguments import get_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from envs.rlbench_env import VoxPoserRLBench
from utils import set_lmp_objects,load_prompt
import numpy as np
from rlbench import tasks
import os
# os.environ['OPENAI_API_KEY'] =  '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' 
import wandb
from PIL import Image


import sys
import json
import argparse
from tqdm import tqdm
import numpy as np


@hydra.main(config_path=f'./configs', config_name="rlbench_config")
def main(cfgs: DictConfig):
    config = cfgs
    # uncomment this if you'd like to change the language model (e.g., for faster speed or lower cost)
    for lmp_name, cfg in config['lmp_config']['lmps'].items():
        cfg['model'] = 'gemini-1.0-pro-latest' #'gpt-4-1106-preview' #'gemini-1.0-pro-latest'/ "gpt-4-1106-preview"

    # initialize env and voxposer ui
    use_server = config['use_server']
    use_sembot = config['use_sembot']

    visualizer = ValueMapVisualizer(config['visualizer'])
    env = VoxPoserRLBench(visualizer=visualizer, save_pcd=False, use_server=use_server,server_ip=config['server_ip'])
    lmps, lmp_env = setup_LMP(env, config, debug=False)
    voxposer_ui = lmps['plan_ui']
    composer_ui = lmps['composer_ui']

    task_list = [
    # 성공률 높음
    # tasks.PutRubbishInBin,
    #  tasks.LampOff,

    # 성공률 약간 낮음
    # tasks.TakeOffWeighingScales,

    # 성공률 매우 낮음
    # tasks.OpenWineBottle,
    # tasks.MeatOffGrill,
    #  tasks.SlideBlockToTarget,
    #  tasks.TakeLidOffSaucepan,
    #  tasks.TakeUmbrellaOutOfUmbrellaStand,
     tasks.PushButton,
    #  tasks.PutGroceriesInCupboard,

    # 아직 task object name도 안만듬
    #  tasks.BeatTheBuzz,
     tasks.ChangeChannel,
     tasks.MoveHanger,
     tasks.ChangeClock,
     tasks.PlugChargerInPowerSupply,

     ]

    plain_dict = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    for task in task_list:
        env.load_task(task)
        task_name = env.task.get_name()
        wandb.init(project="voxposer-naive",
                   entity="pjw971022",
                   name=task_name, 
                   reinit=True)
        wandb.config.update(plain_dict)

        for i in range(cfgs['episode_length']):
            desc_, obs = env.reset()
            if len(desc_)==2:
                descriptions, target_obj = desc_
            else:
                descriptions = desc_
            set_lmp_objects(lmps, env.get_object_names())  # set the object names to be used by voxposer
            instruction = np.random.choice(descriptions) #f"move the {target_obj} to the cupboard." 
            env.instruction = instruction
            print('\n\n\n\n\n\n\n########################## Starting episode #########################')
            print(f'instruction: {instruction}')
            with open('/home/jinwoo/workspace/Sembot/sembot/src/exec_hist.txt', 'a') as f:
                f.write('\n\n\n\n########################## Starting episode #########################\n')
                f.write(f'instruction: {instruction}\n')
            
            wandb.log({"instruction": instruction})
            
            if use_sembot:
                # oracle_plan_code = load_prompt(f"{config['env_name']}/infered_plan/{task_name}.txt")
                # if '{target_obj}' in plan_code:
                #     plan_code = plan_code.replace('{target_obj}', target_obj)
                oracle_plan_code = None
                sembot_cmd = ''
                robot_res = None

                while True:
                    if use_server:
                        sembot_cmd = env.send_obs_server(robot_res)
                    else:
                        sembot_cmd = env.send_obs_local(robot_res)

                    robot_res = None
                    if '[Answer]' in sembot_cmd:
                        break
                    elif '<' in sembot_cmd:
                        print("########  Request for sensor data")
                        robot_res = env.sensor(sembot_cmd)
                    else:
                        print("########  Request for command to composer  ##########")
                        composer_ui(sembot_cmd)
                        robot_res = 'Done.'

                fine_grained_instruction = sembot_cmd.replace('[Answer] ','').strip()
                voxposer_ui(fine_grained_instruction, plan_code=oracle_plan_code) # 
            else:
                voxposer_ui(instruction)
            
    wandb.finish()

if __name__ =='__main__':
    main()