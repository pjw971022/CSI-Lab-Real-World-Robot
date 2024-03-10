import os
import sys
sys.path.append(os.environ['RAVENS_ROOT'])
sys.path.append('/home/pjw971022/Sembot')

from ravens import tasks
from ravens.environments.environment_real import RealEnvironment
from ravens.utils import utils

from real_bot.perception.detection_agent import ObjectDetectorAgent
from real_bot.perception.get_speech import speech_to_command

from custom_utils.llm_utils import *
import numpy as np
import prompts
import os
from prompts.realworld.object_setup import OBJECT_DICT
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
from PIL import Image
import time

@hydra.main(config_path='/home/pjw971022/Sembot/real_bot/rw_config',
            config_name='inference')
def main(cfg):
    env = RealEnvironment(task_name=cfg['task'])
    task = tasks.names[cfg['task']]()
    llm_agent = LLMAgent()
    print(f'@@@ LLM type: {cfg.llm_type} plan mode: {cfg.plan_mode}')
    
    task_name = cfg['task'].replace('real-world-','').replace('-','_')
    agent = ObjectDetectorAgent(task=task_name)
    agent.detector.model.eval()

    step_cnt = 1
    np.random.seed(12)
    env.seed(12)
    env.set_task(task)
    info = env.info
    done = False
    plan_list = None
    env.get_speech()
    final_goal = "imitate my behavior" # speech_to_command()  # "grab the object I'm showing you."
    print("show the image... 5 seconds wait.")
    # time.sleep(5)
    command_modality = 0
    # 0: speech, 1: vision + speech, 2: video + speech
    if 'show' in final_goal: # Pick up the object I'm showing you.
        command_modality = 1
        env.get_obs_human_cam(1)
    elif 'imitate' in final_goal:
        command_modality = 2
        env.get_obs_human_cam(2)

    prompt_cls = prompts.names[cfg['task']]()
    fewshot_prompt = prompt_cls.prompt(command_modality)

    goal_name = final_goal.split(' ')[0]
    objects = OBJECT_DICT[goal_name] #env.task.objects    
    objects_str = ', '.join(objects)
    print("Task setting... 10 seconds wait.")
    # time.sleep(10)
    obs, info_dict = env.reset()

    while not done:
        obs_img = Image.open('/home/pjw971022/Sembot/real_bot/save_vision/obs/image_obs.png')
        ##############################################################################
        if cfg.category_mode == 0: # from LLM 
            if command_modality >=0:
                extract_state_prompt = env.task.extract_state_prompt + f'All possible objects: {objects_str}'
                context_from_vis_obs = llm_agent.gemini_generate_context(extract_state_prompt, obs_img)
                
            if command_modality == 1:
                human_img = Image.open('/home/pjw971022/Sembot/real_bot/save_vision/obs/human_image.png')
                extract_state_prompt = f"[Final goal] {final_goal}. Please change [Final goal] to a command that includes a specific object. Example) pick up the red dice."  + f'All possible objects: {objects_str}'
                final_goal = llm_agent.gemini_generate_context(extract_state_prompt, human_img)

            elif command_modality == 2:
                # human_video = Image.open('/home/pjw971022/Sembot/real_bot/save_vision/obs/human_video.mp4')
                extract_state_prompt = "Please describe the actions of the person appearing in the video. Example) move the toy car to the box"  + f'All possible objects: {objects_str}' + 'Possible Actions: move, pick, place, rotate, push, pull, sweep.'
                human_action = llm_agent.gemini_generate_video_context(extract_state_prompt, 'demo_glasses.mp4')
                final_goal = f"imitate the behavior. [bebavior description] {human_action}" 
            print("final goal: ", final_goal)

        elif cfg.category_mode == 1: # fixed text-context 
            context_from_vis_obs = """
            Brown ring on top of red ring.
            Red ring on top of gray ring.
            Gray ring on top of purple ring.
            Purple ring in lighter brown side.
            The rings can be moved in lighter brown side, middle of the stand, darker brown side.            
            """
            
        print(f"@@@ Context from vision: {context_from_vis_obs}")
        if (cfg.plan_mode == 'open_loop') or step_cnt == 1:
            text_context = f'[Context] {context_from_vis_obs}\n'
            text_context += 'Possible Actions: move, pick, place, rotate, push, pull, sweep.'

            planning_prompt = \
            f'[Goal] {final_goal} {text_context}'

        joined_categories = ", ".join(objects)
        if cfg.plan_mode == 'closed_loop':
            planning_prompt = f'{planning_prompt}' \
                            f'[State of Step {step_cnt}] {joined_categories} '                  
        else:
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
                    plan_list = llm_agent.gemini_gen_all_plan(fewshot_prompt, planning_prompt)
                elif cfg.llm_type == 'gpt4':
                    plan_list = llm_agent.gpt4_gen_all_plan(fewshot_prompt)           
            lang_action = plan_list[step_cnt-1]

        print(f"Plan: {lang_action}")
        if ('done' in lang_action) or ('Done' in lang_action):
            print("Task End!!!")
            break
        elif env.task.max_steps < step_cnt:
            break

        obs['objects'] = objects
        obs['lang_action'] = lang_action
        act = agent.forward(obs)
        
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