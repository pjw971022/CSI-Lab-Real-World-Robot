"""Data collection script."""

import os
import hydra
import numpy as np
import random
import sys
sys.path.append('/home/pjw971022/manipulator_llm/cliport/')

from cliport import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment_real import RealEnvironment


def human_labeling(): # @ 
    pass

@hydra.main(config_path='./cfg', config_name='data')
def main(cfg):
    # Initialize environment and task.
    env = RealEnvironment(
        record_cfg=cfg['record']
    )
    task = tasks.names[cfg['task']]()
    # task.mode = cfg['mode']
    save_data = cfg['save_data']

    # Initialize scripted oracle agent and dataset.
    # agent = task.oracle(env)
    data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], task.mode))
    dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
    print(f"Saving to: {data_path}")
    # print(f"Mode: {task.mode}")

    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = dataset.max_seed
    # Collect training data from oracle demonstrations.
    while dataset.n_episodes < cfg['n']:
        episode, total_reward = [], 0

        env.set_task(task)
        obs = env.reset()
        info = env.info
        reward = 0

        # Rollout expert policy
        for _ in range(task.max_steps):
            # act = agent.act(obs, info)
            lang_goal = info['lang_goal']
            print(f"Language goal: {lang_goal}")
            act = human_labeling()
            episode.append((obs, act, reward, info))
            
            obs, reward, done, info = env.step(act)
            total_reward += reward

            print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
            if done:
                break
        episode.append((obs, None, reward, info))

        # End video recording
        dataset.add(seed, episode)

if __name__ == '__main__':
    main()
