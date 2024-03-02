import os
import sys
sys.path.append(os.environ['RAVENS_ROOT'])
sys.path.append('/c/Users/pjw97/anaconda3/envs/realworld/lib/site-packages')

from ravens import tasks
from ravens.environments.environment_real import RealEnvironment
from ravens.utils import utils

from custom_utils.llm_utils import *
import numpy as np
import prompts
import os
from prompts.realworld.object_setup import OBJECT_DICT
os.environ["TOKENIZERS_PARALLELISM"] = "false"

        
import hydra
from PIL import Image
import sys
sys.path.append('/c/Users/pjw97/workspace/RealWorldLLM')
import time
from perception.detection_agent import ObjectDetectorAgent
from perception.sst import speech_to_command

@hydra.main(config_path='/home/pjw971022/RealWorldLLM/rw_config',
            config_name='inference')
def main(cfg):
    env = RealEnvironment(task_name=cfg['task'])
    task = tasks.names[cfg['task']]()
    # lamorel_args = cfg.lamorel_args
    llm_agent = LLMAgent(cfg.use_vision_fewshot)
    print(f'@@@ LLM type: {cfg.llm_type}   vision fewshot: {cfg.use_vision_fewshot}   plan mode: {cfg.plan_mode}')
    
    if cfg.agent_mode==3:
        prompt_cls = prompts.names[cfg['task']]()
        if cfg.llm_type !='no':
            if cfg.command_format in ['language','speech']:
                fewshot_prompt = prompt_cls.prompt()

            task_name = cfg['task'].replace('real-world-','').replace('-','_')
            # fewshot_img = Image.open(f'/home/pjw971022/RealWorldLLM/save_viz/obs/{task_name}_fewshot_img.png')
        print(f"Task: {task_name}")
        agent = ObjectDetectorAgent(task=task_name)
        agent.detector.model.eval()

    # for i in range(cfg.eval_episodes):
    step_cnt = 1
    np.random.seed(12)
    env.seed(12)
    env.set_task(task)
    obs, info_dict = env.reset()
    info = env.info
    done = False
    plan_list = None
    if 'speech' in cfg['task']:
        final_goal = speech_to_command()# 'take out all the items from the basket.' #  #'take out all the items from the basket.' # #'clean up the every ball.' # 
        #################################
        # take out all the items from the basket. 
        # clear the spilled cola.
        # pack the green objects.
        # prepare the meal, please.    
        # play with the toy car. 
        # clean up the every ball.
        # speech_to_command()
    else:
        final_goal = info['final_goal']
        
    goal_name = final_goal.split(' ')[0]
    objects = OBJECT_DICT[goal_name] #env.task.objects    
    objects_str = ', '.join(objects)
    
    while not done:
        obs_img = Image.open('/home/pjw971022/RealWorldLLM/save_viz/obs/image_obs.png')
        ##############################################################################
        if cfg.category_mode == 0: # from LLM 
            extract_state_prompt = env.task.extract_state_prompt + f'All possible objects: {objects_str}'
            context_from_vision = llm_agent.gpt4_generate_context(extract_state_prompt, obs_img)
            print(f"@@@ Context from vision: {context_from_vision}")
            if (cfg.plan_mode == 'open_loop') or step_cnt == 1:
                if cfg.command_format in ['speech', 'language'] :
                    text_context = f'[Context] {context_from_vision}'
                    text_context +=  f'All possible objects: {objects_str}'
                    text_context += 'Possible Actions: move, pick, place, rotate, push, pull, sweep.'
                                    
                    planning_prompt = \
                    f'[Goal] {final_goal}. {text_context}'
                else:
                    planning_prompt = ''
        joined_categories = ", ".join(objects)
        if cfg.command_format == 'language':
            planning_prompt = f'{planning_prompt}' \
                            f'[State of Step {step_cnt}] {joined_categories} '                  
        elif cfg.command_format == 'speech':
            planning_prompt = f'{planning_prompt}'
        ##############################################################################
        if cfg.plan_mode == 'closed_loop':
            planning_prompt +=  f'[Plan {step_cnt}] '
            if cfg.llm_type == 'gemini':
                gen_act = llm_agent.gemini_gen_act(fewshot_prompt, planning_prompt, obs_img)
            elif cfg.llm_type == 'palm':
                gen_act = llm_agent.palm_gen_act(fewshot_prompt, planning_prompt)
            elif cfg.llm_type == 'gpt4':
                gen_act = llm_agent.gpt4_gen_act(fewshot_prompt, planning_prompt)
            lang_action = gen_act

        elif cfg.plan_mode == 'open_loop':
            if step_cnt == 1:
                if cfg.llm_type == 'gemini':
                    plan_list = llm_agent.gemini_gen_all_plan(fewshot_prompt, planning_prompt, obs_img)
                elif cfg.llm_type == 'gpt4':
                    plan_list = llm_agent.gpt4_gen_all_plan(fewshot_prompt)
                    
            lang_action = plan_list[step_cnt-1]
        print(f"Plan: {lang_action}")
        if ('done' in lang_action) or ('Done' in lang_action):
            print("Task End!!!")
            break

        elif env.task.max_steps < step_cnt:
            break

        if cfg.agent_mode==3:
            obs['objects'] = objects
            obs['lang_action'] = lang_action
            act = agent.forward(obs)

        elif cfg.agent_mode==2:
            info['lang_goal'] = lang_action
            env.task.lang_goals = [lang_action]
            act = agent.act(obs, info, None)
        
        if act == -1:
            continue

        z = env.step(act)
        try:
            obs, reward, done, info = z
        except ValueError:
            obs, reward, done, info, action_ret_str = z
            print(action_ret_str)


        planning_prompt = f'{planning_prompt}{lang_action}. '
        step_cnt += 1

if __name__ == "__main__":
    main()