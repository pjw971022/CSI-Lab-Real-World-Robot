import os
import sys
sys.path.append(os.environ['RAVENS_ROOT'])
from ravens import tasks
from ravens.environments.environment_real import RealEnvironment
from ravens.utils import utils

from custom_utils.llm_utils import *
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
import hydra
import sys
sys.path.append('/home/pjw971022/Sembot/cliport/')

from real_bot.perception.detection_agent import ObjectDetectorAgent
@hydra.main(config_path='/home/pjw971022/Sembot/real_bot/rw_config',
            config_name='inference')
def main(cfg):
    env = RealEnvironment()
    pick_obj = 'plastic container'
    agent = ObjectDetectorAgent(cfg['task'], pick_obj =pick_obj)
    agent.detector.model.eval()
    
    env.set_task(cfg['task'])
    obs = env.reset()

    lang_action = f'move the {pick_obj} in the green box.' \
                  f'[Detailed Guideline 1] Grab position: the edge of objects' \
                  f'[Detailed Guideline 2] Gripper Orientation: vertical' 
    

    obs['pick_objects'] = [f'{pick_obj}']
    obs['lang_action'] = lang_action
    
    act = agent.forward(obs)

    z = env.step(act)
    try:
        obs, reward, done, info = z
    except ValueError:
        obs, reward, done, info, action_ret_str = z
        print(action_ret_str)

if __name__ == "__main__":
    main()