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
    # info = env.info
    done = False
    plan_list = None
    env.get_speech()
    final_goal = "imitate my behavior" # speech_to_command()  # "grab the object I'm showing you."
    
    demo_context = None
    obs_context = None
    command_modality = 0
    # 0: speech, 1: vision + speech, 2: video + speech #: video + image + speech
    if 'show' in final_goal: # Pick up the object I'm showing you.
        print("show the image ... 10 seconds wait.")
        env.send_display_server(1, 'start')
        time.sleep(10)
        env.send_display_server(1, 'end')
        command_modality = 1
        env.get_obs_human_cam(1)
        
    elif 'imitate' in final_goal:
        print("show the video ... 10 seconds wait.")
        env.send_display_server(1, 'start')
        time.sleep(25)
        env.send_display_server(1, 'end')
        command_modality = 2
        env.get_obs_human_cam(2)

    prompt_cls = prompts.names[cfg['task']]()
    fewshot_prompt = prompt_cls.prompt(command_modality)
    env.send_display_server(2, 'start')
    time.sleep(10)
    env.send_display_server(2, 'end')
    
    goal_name = final_goal.split(' ')[0]
    objects = OBJECT_DICT[goal_name] #env.task.objects    
    objects_str = ', '.join(objects)
    env.send_display_server(3, 'start')
    obs, info_dict = env.reset()
    obs_img = Image.open(f'/home/pjw971022/Sembot/real_bot/save_vision/obs/image_obs_{step_cnt}.png')

    if command_modality == 1:
        prompt_extract_state = env.task.prompt_extract_state + f'All possible objects: {objects_str}'
        obs_context = llm_agent.gemini_generate_context(prompt_extract_state, obs_img)
    
        human_img = Image.open('/home/pjw971022/Sembot/real_bot/save_vision/obs/human_image.png')
        prompt_extract_goal = f"[Final goal] {final_goal}. Please change [Final goal] to a command that includes a specific object. Example) pick up the red dice."  + f'All possible objects: {objects_str}'
        final_goal = llm_agent.gemini_generate_context(prompt_extract_goal, human_img)

    elif command_modality == 2:
        prompt_extract_demo = "Please explain the human behavior in this video."
        prompt_extract_demo += f'\nAll possible objects: {objects_str}'
        demo_context = llm_agent.gemini_generate_video_context(prompt_extract_demo, 'demo_glasses.mp4')

    elif command_modality == 3:
        prompt_extract_state = env.task.prompt_extract_state + f'All possible objects: {objects_str}'
        obs_context = llm_agent.gemini_generate_context(prompt_extract_state, obs_img)
        
        prompt_extract_demo = f"Please describe the actions of the person appearing in the video. {fewshot_prompt}"  + f'All possible objects: {objects_str}' + 'Possible Actions: move, pick, place, rotate, push, pull, sweep.'
        demo_context = llm_agent.gemini_generate_video_context(prompt_extract_demo, 'demo_glasses.mp4')

        human_img = Image.open('/home/pjw971022/Sembot/real_bot/save_vision/obs/human_image.png')
        prompt_extract_goal = f"[Final goal] {final_goal}. Please change [Final goal] to a command that includes a specific object. Example) pick up the red dice."  + f'All possible objects: {objects_str}'
        final_goal = llm_agent.gemini_generate_context(prompt_extract_goal, human_img)

    while not done:
        obs_img = Image.open(f'/home/pjw971022/Sembot/real_bot/save_vision/obs/image_obs_{step_cnt}.png')
        if cfg.plan_mode == 'closed_loop': # from LLM 
            prompt_extract_state = env.task.prompt_extract_state + f'All possible objects: {objects_str}'
            obs_context = llm_agent.gemini_generate_context(prompt_extract_state, obs_img)
                
        ##############################################################################
        if (cfg.plan_mode == 'open_loop') and step_cnt == 1:
            text_context = '[Context] '
            if demo_context is not None:
                print("demo_context: ", demo_context)
                text_context += f'{demo_context}\n'
            if obs_context is not None:
                print(f"@@@ Context from vision: {obs_context}")
                text_context += f'{obs_context}\n' 
            text_context += f'All possible objects: {objects_str}'
            text_context += 'Possible Actions: move, pick, place, rotate, push, pull, sweep.'
            planning_prompt = \
            f'[Goal] {final_goal} {text_context}'
        
        if cfg.plan_mode == 'closed_loop':
            joined_categories = ", ".join(objects)
            planning_prompt = f'{planning_prompt}' \
                            f'[State of Step {step_cnt}] {joined_categories} '                  
        else:
            planning_prompt = f'{planning_prompt}'

        ##############################################################################
        if cfg.plan_mode == 'closed_loop':
            planning_prompt +=  f'[Plan {step_cnt}] '
            if cfg.llm_type == 'gemini':
                gen_act = llm_agent.gemini_gen_act(fewshot_prompt, planning_prompt, obs_img)
            elif cfg.llm_type == 'gpt4':
                gen_act = llm_agent.gpt4_gen_act(fewshot_prompt, planning_prompt)
            lang_action = gen_act

        elif cfg.plan_mode == 'open_loop' and step_cnt == 1:
            if plan_list == None:
                if cfg.llm_type == 'gemini':
                    plan_list = llm_agent.gemini_gen_all_plan(fewshot_prompt, planning_prompt)
                elif cfg.llm_type == 'gpt4':
                    plan_list = llm_agent.gpt4_gen_all_plan(fewshot_prompt, planning_prompt)
            lang_action = plan_list[step_cnt-1]
        else:
            plan_list= ["pull the <first drawer handle>", #"push the <drawer handle 2>",
                        "pull the <second drawer handle>", #"push the <drawer handle 4>", 
                        "pick up the <coffee stick>",
                        "place in the <setting point1>",
                        "move the <red cup> to the <setting point2>",
                        "done imitating your behavior."]
            
            lang_action = plan_list[step_cnt-1]
        print(f"Plan: {lang_action}")
        if ('done' in lang_action) or ('Done' in lang_action):
            print("Task End!!!")
            env.send_display_server(4, 'end')
            break
        elif env.task.max_steps < step_cnt:
            break

        obs['objects'] = objects
        obs['lang_action'] = lang_action
        obs['step_cnt'] = step_cnt
        act = agent.forward(obs)
        step_cnt += 1
        env.send_display_server(3, 'end')

        if act == -1:
            continue
        if step_cnt == 1:
            env.send_display_server(4, 'start')

        # Continue with the rest of the code
        z = env.step(act)
        try:
            obs, reward, done, info = z
        except ValueError:
            obs, reward, done, info, action_ret_str = z
            print(action_ret_str)

        planning_prompt = f'{planning_prompt}{lang_action}. '
    
if __name__ == "__main__":
    main()