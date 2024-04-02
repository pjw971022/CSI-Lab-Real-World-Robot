import hydra
from omegaconf import DictConfig
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
        cfg['model'] = 'gemini-1.0-pro-latest'

    # initialize env and voxposer ui
    visualizer = ValueMapVisualizer(config['visualizer'])
    env = VoxPoserRLBench(visualizer=visualizer)
    lmps, lmp_env = setup_LMP(env, config, debug=False)
    voxposer_ui = lmps['plan_ui']
    task_list = [tasks.PutRubbishInBin,
     tasks.LampOff,
     tasks.OpenWineBottle,
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
    wandb.init(project="voxposer-naive", entity="pjw971022", name=config.context_mode)
    wandb.config.update(config)
    for task in task_list:
        for i in range(cfg['episode_length']):
            env.load_task(task)
            descriptions, obs = env.reset()
            set_lmp_objects(lmps, env.get_object_names())  # set the object names to be used by voxposer
            task_name = env.task.get_name()
            instruction = np.random.choice(descriptions)
            print(f'instruction: {instruction}')
            voxposer_ui(instruction)
        # Log metrics to wandb
        wandb.log({"task_name": task_name})
    wandb.finish()

if __name__ =='__main__':
    main()