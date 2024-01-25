import os
import sys
sys.path.append(os.environ['RAVENS_ROOT'])
from ravens import tasks
from ravens.environments.environment import Environment

from custom_utils.llm_utils import *
import numpy as np
import prompts
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
def parse_action(lang_action, task):
    """ parse action to retrieve pickup object and place object"""
    lang_action = re.sub(r'[^\w\s]', '', lang_action)  # remove all strings
    if task == 'put-block-in-bowl':
        target_pattern = r'([a-zA-Z]+ block \d+)'
        recep_pattern = r'([a-zA-Z]+ bowl \d+)'
    elif task == 'towers-of-hanoi-seq':
        target_pattern = r'([a-zA-Z]+ disk)'
        recep_pattern = r"(rod 1|rod 2|rod 3)"
    elif task == 'towers-of-hanoi-seq-seen':
        target_pattern = r'([a-zA-Z]+ ring)'
        recep_pattern = r"(lighter brown side|middle of the stand|darker brown side)"

    else:
        raise NotImplementedError
    target_match = re.search(target_pattern, lang_action)
    recep_match = re.search(recep_pattern, lang_action)  # receptacle

    if target_match and recep_match:
        target = target_match.group(1)
        recep = recep_match.group(1)
        return target, recep
    else:
        return None, None
        
import hydra
from PIL import Image
import sys
sys.path.append('/home/pjw971022/manipulator_llm/cliport/')
from cliport import agents
from cliport.utils import utils
from cliport.tasks import cameras
from cliport import dataset
import datetime
from cliport.environment_real import RealEnvironment
CAMERA_CONFIG = cameras.RealSenseD415.CONFIG

def get_image(obs, cam_config=None):
    """Stack color and height images image."""
    bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
    pix_size = 0.003125
    in_shape = (320, 160, 6) # @ 
    # if self.use_goal_image:
    #   colormap_g, heightmap_g = utils.get_fused_heightmap(goal, configs)
    #   goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
    #   input_image = np.concatenate((input_image, goal_image), axis=2)
    #   assert input_image.shape[2] == 12, input_image.shape

    if cam_config is None:
        cam_config = CAMERA_CONFIG

    # Get color and height maps from RGB-D images.
    cmap, hmap = utils.get_fused_heightmap(
        obs, cam_config, bounds, pix_size)
    img = np.concatenate((cmap,
                            hmap[Ellipsis, None],
                            hmap[Ellipsis, None],
                            hmap[Ellipsis, None]), axis=2)
    assert img.shape == in_shape, img.shape
    return img


@hydra.main(config_path=os.path.join(os.environ['RAVENS_ROOT'], 'ravens/cfg'),
            config_name='inference')
def main(cfg):
    os.environ['CKPT_ROOT'] = \
    os.path.join(os.environ['RAVENS_ROOT'], cfg['task'], 'checkpoints')

    env = RealEnvironment(
        record_cfg=cfg['record']
    )
        
    lamorel_args = cfg.lamorel_args
    llm_agent = LLMAgent(lamorel_args, cfg.llm_type)

    # few_shot_prompt = prompts.names['put-block-in-bowl']().prompt()
    # import ipdb;ipdb.set_trace()
    domain = tasks.names[cfg['task']](task_level=cfg['task_level'])
    domain.mode = cfg['mode']
    record = cfg['record']['save_video']
    if cfg.agent_mode==0:
        agent = domain.oracle(env)
        few_shot_prompt = ''
    
    elif cfg.agent_mode==1:
        agent = domain.langAgent(env)
        # prompting for llms
        prompt_cls = prompts.names[cfg['task']]()
        few_shot_prompt = prompt_cls.prompt()
    elif cfg.agent_mode==2:
        prompt_cls = prompts.names[cfg['task']](n_shot=3)
        few_shot_prompt = prompt_cls.prompt()
        model_file = os.path.join(cfg['cliport_model_path'], 'last.ckpt')
        # Initialize agent.
        # utils.set_seed(train_run, torch=True)
        tcfg = utils.load_hydra_config(cfg['train_config'])
        ds = dataset.RavensDataset(os.path.join(cfg['data_dir'], f"{cfg['task']}-{cfg['mode']}"),
                                    tcfg,
                                    n_demos=cfg['n_demos'],
                                    augment=False)
        agent = agents.names[cfg['agent']]('cliport', tcfg, None, ds)

        # Load checkpoint
        agent.load(model_file)
        print(f"Loaded: {model_file}")

    for i in range(cfg.eval_episodes):
        step_cnt = 1
        np.random.seed(12)
        env.seed(12)
        env.set_task(domain)
        obs = env.reset()
        info = env.info
        done = False
        state_lang = env.task.lang_initial_state
        lang_goal = info['lang_goal']
        # possible_actions = info['possible_actions']
        admissible_actions_list = env.task.admissible_actions
        task_completed_desc = env.task.task_completed_desc
        admissible_actions_list.append(task_completed_desc)

        prompt = \
        f'{few_shot_prompt}' \
        f'[Goal] {lang_goal}. ' \
        f'[Initial State] {state_lang} ' \
        f'[Step 1] '
        print(f"@@@ GOAL: {lang_goal}")
        print(f"@@@ Initial State: {state_lang}")
        # if record:
        #     updated_video_name = f"{cfg['task']}---{lang_goal}"# Update the video name
        #     env.start_rec(updated_video_name)  # Start a new recording
        while not done:
            ########################  LLM Scoring #####################
            if cfg.llm_type == 'open':
                llm_score, gen_act = llm_agent.openllm_new_scoring(prompt, admissible_actions_list)
            elif cfg.llm_type == 'palm':
                llm_score, gen_act = llm_agent.palm_new_scoring(prompt, admissible_actions_list)
            elif cfg.llm_type == 'gpt':
                llm_score = llm_agent.gpt3_scoring(prompt, admissible_actions_list) 

            lang_action = max(llm_score, key=llm_score.get)
            if cfg.agent_mode ==2:
                info['lang_goal'] = lang_action
                env.task.lang_goals = [lang_action]
                act = agent.act(obs, info, None)
            else:
                act = agent.act(parse_action(lang_action=lang_action,
                                                task=cfg['task']),
                                    obs, info)  # pick_pose, place_pose
            
            z = env.step(act)
            try:
                obs, reward, done, info = z
            except ValueError:
                obs, reward, done, info, action_ret_str = z
                print(action_ret_str)
            
            if step_cnt > cfg.max_steps:
                print("timeout")
                break
            if 'done putting disks in rods'==lang_action:
                break
            print(f"reward: {reward} done: {done}")
            print("\n")

            prompt = f'{prompt}{lang_action}. [Step {step_cnt+1}] '
            step_cnt += 1

          
# visualize 
            
if __name__ == "__main__":
    main()