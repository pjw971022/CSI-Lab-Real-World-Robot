"""Data collection script."""

import os
import hydra
import numpy as np

from ravens import tasks
from ravens.dataset import RavensDataset
from ravens.environments.environment_real import RealEnvironment
from PIL import Image
import matplotlib.pyplot as plt


BOUND_X = (0.2, 0.8)
BOUND_Y = (-0.5, 0.7)
# e.g
# Pack the green dice in the green basket
# Pack the red dice in the green basket
# Pack the yellow dice in the green basket
# Pack the tennis ball dice in the green basket
# Pack the baseball dice in the green basket
import tkinter as tk
from PIL import Image, ImageTk
def transform_coordinates(x, y):
    # 픽셀 좌표 (290, 150)이 새 좌표 (0.55, -0.1)와 일치하도록 변환 비율을 설정합니다.
    scale_x = 1 / 1200
    scale_y = 1 / 1200

    # 새 좌표 시스템에서의 좌표를 계산합니다.
    new_x = (x - 410) * scale_x
    new_y = (y + 390)* scale_y
    return new_x, new_y

def find_coordinate_interface(image):
    new_x, new_y = None, None
    def on_click(event):
        new_x, new_y = transform_coordinates(event.x, event.y)
        print("Clicked at: x =", new_x, ", y =", new_y)

    root = tk.Tk()
    root.title("Click Coordinates")

    photo = ImageTk.PhotoImage(image)

    label = tk.Label(root, image=photo)
    label.pack()

    label.bind("<Button-1>", on_click)
    root.mainloop()
    return new_x, new_y

def human_labeling(image): # @ 
    lang_goal = input("Enter your language goal: ")
    print("Click Pick Coordinate")
    pick_x,  pick_y = find_coordinate_interface(image)
    
    print("Click Place Coordinate")
    place_x,  place_y = find_coordinate_interface(image)
    
    assert (pick_x > BOUND_X[0]) and (pick_x < BOUND_X[1])
    assert (place_x > BOUND_X[0]) and (place_x < BOUND_X[1])
    assert (pick_y > BOUND_Y[0]) and (pick_y < BOUND_Y[1])
    assert (place_y > BOUND_Y[0]) and (place_y < BOUND_Y[1])
    
    return (pick_x, pick_y, 0.08, 0, np.pi, 0), (place_x, place_y, 0.08, 0, np.pi, 0), lang_goal

def grid_overlay(img, pp_cnt):
    plt.imshow(img)

    plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)

    plt.xlabel("X-coordinate in pixels")
    plt.ylabel("Y-coordinate in pixels")

    plt.title("Image with Grid Overlay")
    plt.savefig(f'/home/pjw971022/Sembot/real_bot/save_vision/demo_viz/obs_{pp_cnt}.png')

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
            pose0, pose1, lang_goal = human_labeling(image)
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
