"""Data collection script."""

import os
import hydra
import numpy as np
import random
import sys
sys.path.append('/home/pjw971022/RealWorldLLM/cliport/')

from cliport import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment_real import RealEnvironment
from PIL import Image
BOUND_X = (0.2, 0.8)
BOUND_Y = (-0.5, 0.7)
# e.g
# Pack the green dice in the green basket
# Pack the red dice in the green basket
# Pack the yellow dice in the green basket
# Pack the tennis ball dice in the green basket
# Pack the baseball dice in the green basket

def human_labeling(): # @ 
    lang_goal = input("Enter your language goal: ")
    pick_x = float(input("Enter pick x coordinate: "))
    pick_y = float(input("Enter pick y coordinate: "))

    place_x = float(input("Enter pick x coordinate: "))
    place_y = float(input("Enter pick y coordinate: "))
    
    assert (pick_x > BOUND_X[0]) and (pick_x < BOUND_X[1])
    assert (place_x > BOUND_X[0]) and (place_x < BOUND_X[1])
    assert (pick_y > BOUND_Y[0]) and (pick_y < BOUND_Y[1])
    assert (place_y > BOUND_Y[0]) and (place_y < BOUND_Y[1])
    
    return (pick_x, pick_y, 0.07, 0, np.pi, 0), (place_x, place_y, 0.07, 0, np.pi, 0), lang_goal

@hydra.main(config_path='./cfg', config_name='data')
def main(cfg):
    # Initialize environment and task.
    env = RealEnvironment()
    task = tasks.names[cfg['task']]()
    # task.mode = cfg['mode']
    save_data = cfg['save_data']

    # Initialize scripted oracle agent and dataset.
    # agent = task.oracle(env)
    data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], task.mode))
    dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
    print(f"Saving to: {data_path}")

    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = dataset.max_seed
    # Collect training data from oracle demonstrations.
    while dataset.n_episodes < cfg['n']:
        episode, total_reward = [], 0
        env.set_task(task)
        obs = env.reset()
        info = env.info
        reward = 0
        pp_cnt = 0
        print("Reset Real-world Environment")
        while pp_cnt < 5:
            # Rollout expert policy
            image =Image.fromarray(obs['color'][0].astype(np.uint8))
            image.save(f'/home/pjw971022/RealWorldLLM/save_viz/demo_viz/obs_{pp_cnt}.png')
            pose0,pose1, lang_goal = human_labeling()
            act = {'pose0':pose0, 'pose1':pose1}
            info['lang_goal'] = lang_goal
            print(f"Language goal: {lang_goal}")
            episode.append((obs, act, reward, info))
            
            obs, reward, _, info = env.step(act, True)
            
            check_success = input("Was pick & place successful? [Y/n]")
            if check_success=='Y':
                reward = 1
            else:
                reward = 0
            _, _, _, _ = env.step(2)
            total_reward += reward
            print(f'Total Reward: {total_reward:.3f} | Goal: {lang_goal}')

            episode.append((obs, None, reward, info))
        dataset.add(seed, episode)

if __name__ == '__main__':
    main()
