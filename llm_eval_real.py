import os
import sys
sys.path.append(os.environ['RAVENS_ROOT'])
from ravens import tasks
from ravens.environments.environment_real import RealEnvironment
from ravens.utils import utils

from custom_utils.llm_utils import *
import numpy as np
import prompts
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

        
import hydra
from PIL import Image
import sys
sys.path.append('/home/pjw971022/RealWorldLLM/cliport/')


import time

from open_vocab.detection_agent import ObjectDetectorAgent
@hydra.main(config_path='/home/pjw971022/RealWorldLLM/rw_config',
            config_name='inference')
def main(cfg):
    env = RealEnvironment()
    task = tasks.names[cfg['task']]()
    # lamorel_args = cfg.lamorel_args
    llm_agent = LLMAgent(cfg.use_vision_fewshot)
    print(f'@@@ LLM type: {cfg.llm_type}   vision fewshot: {cfg.use_vision_fewshot}   plan mode: {cfg.plan_mode}')
    # if cfg.agent_mode==2:
    #     prompt_cls = prompts.names[cfg['task']]()
    #     fewshot_prompt = prompt_cls.prompt()
    #     model_file = os.path.join(cfg['cliport_model_path'], 'last.ckpt')
    #     tcfg = utils.load_hydra_config(cfg['train_config'])
    #     agent = agents.names[cfg['agent']]('cliport', tcfg, None, None)

    #     # Load checkpoint
    #     agent.load(model_file)
    #     print(f"Loaded: {model_file}")
    
    if cfg.agent_mode==3:
        prompt_cls = prompts.names[cfg['task']]()
        if cfg.llm_type !='no':
            if cfg.command_format == 'language' :
                fewshot_prompt = prompt_cls.prompt()
            else:
                fewshot_prompt = prompt_cls.video2prompt()
            task_name = cfg['task'].replace('real-world-','').replace('-','_')
            fewshot_img = Image.open(f'/home/pjw971022/RealWorldLLM/save_viz/obs/{task_name}_fewshot_img.png')
        
        agent = ObjectDetectorAgent()
        agent.detector.model.eval()

    for i in range(cfg.eval_episodes):
        step_cnt = 1
        np.random.seed(12)
        env.seed(12)
        env.set_task(task)
        obs = env.reset()
        info = env.info
        done = False
        plan_list = None
        final_goal = info['final_goal']
        receptacles = env.receptacles
        if cfg.command_format=='language':
            planning_prompt = \
            f'[Goal] {final_goal}. '
        else:
            planning_prompt = ''
                
        while not done:
            obs_img = Image.open('/home/pjw971022/RealWorldLLM/save_viz/obs/image_obs.png')
            if cfg.category_mode == 0: # from LLM 
                extract_state_prompt = env.task.extract_state_prompt
 
                objects = llm_agent.gemini_generate_categories(extract_state_prompt, obs_img)
                print(f"@@@ Categories: {objects}")

                admissible_actions_list = [f'move the {obj} in the {recep}' for obj in objects for recep in receptacles]
                task_completed_desc = env.task.task_completed_desc
                admissible_actions_list.append(task_completed_desc)
            else:
                objects = env.task.objects
                admissible_actions_list = env.task.admissible_actions
                task_completed_desc = env.task.task_completed_desc
                admissible_actions_list.append(task_completed_desc)
            joined_categories = ", ".join(objects)
            print(f'objects: {objects}')
            if cfg.command_format == 'language':
                planning_prompt = f'{planning_prompt}' \
                                f'[State of Step {step_cnt}] {joined_categories} ' \
                                f' [Plan {step_cnt}] '

            if cfg.plan_mode == 'closed_loop':
                if cfg.llm_type == 'gemini':
                    gen_act = llm_agent.gemini_gen_act(fewshot_prompt, planning_prompt, fewshot_img, obs_img)
                elif cfg.llm_type == 'palm':
                    gen_act = llm_agent.palm_gen_act(fewshot_prompt, planning_prompt)
                elif cfg.llm_type == 'gpt4':
                    gen_act = llm_agent.gpt4_gen_act(fewshot_prompt, planning_prompt)
                lang_action = gen_act
            elif cfg.plan_mode == 'open_loop':
                if step_cnt == 1:
                    if cfg.llm_type == 'gpt4':
                        plan_list = llm_agent.gpt4_gen_all_plan(fewshot_prompt)
                
                lang_action = plan_list[step_cnt-1]

            print(f"Plan: {lang_action}")
            
            if ('done' in lang_action) or ('Done' in lang_action):
                break
            elif env.task.max_steps < step_cnt:
                break

            if cfg.agent_mode==3:
                obs['pick_objects'] = objects
                obs['lang_action'] = lang_action
                
                act = agent.forward(obs)
                if act == -1:
                    lang_action += '[Failure] '
                    planning_prompt = f'{planning_prompt}{lang_action}. '
                    step_cnt += 1
                    continue
                
            elif cfg.agent_mode==2:
                info['lang_goal'] = lang_action
                env.task.lang_goals = [lang_action]
                act = agent.act(obs, info, None)

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