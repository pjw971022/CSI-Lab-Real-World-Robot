# set GPU
# %env CUDA_VISIBLE_DEVICES=0
# %env RAVENS_ROOT=/home/pjw971022/RealWorldLLM/ravens/
import os
import sys
import json
import sys
sys.path.append('/home/pjw971022/RealWorldLLM/cliport/')
import numpy as np
from cliport import tasks
from cliport import agents
from cliport.utils import utils
from cliport.tasks import cameras
from cliport.environments.environment_real import RealEnvironment
import torch
# sys.path.append(os.environ['RAVENS_ROOT'])
# from ravens import tasks
# from ravens.environments.environment import Environment
# from ravens.dataset import RavensDataset
import numpy as np
import matplotlib.pyplot as plt
CAMERA_CONFIG = cameras.RealSenseD435.CONFIG
def get_image(obs, cam_config=None):
    """Stack color and height images image."""
    bounds = np.array([[0.2, 0.8], [-0.5, 0.7], [-0.2, 0.3]]) # @
    #bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
    pix_size = 0.003125 * 1.2
    in_shape = (320, 160, 6) # @ 
    # if self.use_goal_image:
    #   colormap_g, heightmap_g = utils.get_fused_heightmap(goal, configs)
    #   goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
    #   input_image = np.concatenate((input_image, goal_image), axis=2)
    #   assert input_image.shape[2] == 12, input_image.shape

    if cam_config is None:
        cam_config = CAMERA_CONFIG

    # Get color and height maps from RGB-D images.
    cmap, hmap = utils.get_fused_heightmap_real(
        obs, cam_config, bounds, pix_size) 
    img = np.concatenate((cmap,
                            hmap[Ellipsis, None],
                            hmap[Ellipsis, None],
                            hmap[Ellipsis, None]), axis=2)
    assert img.shape == in_shape, img.shape
    return img


train_demos = 1000 # number training demonstrations used to train agent
n_eval = 1 # number of evaluation instances
mode = 'test' # val or test

agent_name = 'cliport'
model_task = 'multi-language-conditioned' # multi-task agent conditioned with language goals

model_folder = 'cliport_quickstart' # path to pre-trained checkpoint
ckpt_name = 'last.ckpt' # name of checkpoint to load

draw_grasp_lines = True
affordance_heatmap_scale = 30
eval_task = 'real-world-1'

root_dir = '/home/pjw971022/RealWorldLLM/cliport/'
assets_root = os.path.join(root_dir, 'cliport/environments/assets/')
config_file = 'eval.yaml' 

vcfg = utils.load_hydra_config(os.path.join(root_dir, f'cliport/cfg/{config_file}'))
vcfg['data_dir'] = '/home/mnt/data/ravens/'
vcfg['mode'] = mode

vcfg['model_task'] = model_task
vcfg['eval_task'] = eval_task
vcfg['agent'] = agent_name

# Model and training config paths
model_path = os.path.join(root_dir, model_folder)
vcfg['train_config'] = f"{model_path}/{vcfg['model_task']}-{vcfg['agent']}-n{train_demos}-train/.hydra/config.yaml"
vcfg['model_path'] = f"{model_path}/{vcfg['model_task']}-{vcfg['agent']}-n{train_demos}-train/checkpoints/"

tcfg = utils.load_hydra_config(vcfg['train_config'])

# Load dataset
# ds = RavensDataset(os.path.join(vcfg['data_dir'], f'{vcfg["eval_task"]}-{vcfg["mode"]}'), 
#                    tcfg, 
#                    n_demos=n_eval,
#                    augment=False)

eval_run = 0
name = '{}-{}-{}-{}'.format(vcfg['eval_task'], vcfg['agent'], n_eval, eval_run)
print(f'\nEval ID: {name}\n')

# Initialize agent
utils.set_seed(eval_run, torch=True)
agent = agents.names[vcfg['agent']](name, tcfg, None, None)

# Load checkpoint
ckpt_path = os.path.join(vcfg['model_path'], ckpt_name)
print(f'\nLoading checkpoint: {ckpt_path}')
agent.load(ckpt_path)


# Initialize environment and task.
env = RealEnvironment(
)


episode = 0
num_eval_instances = 1 # min(n_eval, ds.n_episodes)

for i in range(num_eval_instances):
    print(f'\nEvaluation Instance: {i + 1}/{num_eval_instances}')
    
    # Load episode
    # episode, seed = ds.load(i)
    seed = 0
    goal = None #episode[-1]
    total_reward = 0
    np.random.seed(seed)
    
    # Set task
    task_name = vcfg['eval_task']
    print("Task name: ", task_name)
    # print(f"State:{env.task.lang_initial_state}")
    task = tasks.names[task_name]()
    task.mode = mode
    
    # Set environment
    env.seed(seed)
    env.set_task(task)
    obs = env.reset()
    info = env.info
    reward = 0
    
    step = 0
    done = False
    image_data = obs['color'][0].astype(np.uint8)
    plt.imsave('/home/pjw971022/RealWorldLLM/save_viz/cliport_only/heatmap/rgb_init.png', image_data, cmap='gray')

    # Rollout
    while (step < task.max_steps) and not done:
        print(f"Step: {step} ({task.max_steps} max)")
        fig, axs = plt.subplots(2, 2, figsize=(13, 7))
        
        # Get color and depth inputs
        img = get_image(obs) # batch['img']
        img = torch.from_numpy(img)
        color = np.uint8(img.detach().cpu().numpy())[:,:,:3]
        color = color.transpose(1,0,2)
        depth = np.array(img.detach().cpu().numpy())[:,:,3]
        depth = depth.transpose(1,0)
        
        # Display input color
        axs[0,0].imshow(color)
        axs[0,0].axes.xaxis.set_visible(False)
        axs[0,0].axes.yaxis.set_visible(False)
        axs[0,0].set_title('Input RGB')
        
        # Display input depth
        axs[0,1].imshow(depth)
        axs[0,1].axes.xaxis.set_visible(False)
        axs[0,1].axes.yaxis.set_visible(False)        
        axs[0,1].set_title('Input Depth')
        
        # Display predicted pick affordance
        axs[1,0].imshow(color)
        axs[1,0].axes.xaxis.set_visible(False)
        axs[1,0].axes.yaxis.set_visible(False)
        axs[1,0].set_title('Pick Affordance')
        
        # Display predicted place affordance
        axs[1,1].imshow(color)
        axs[1,1].axes.xaxis.set_visible(False)
        axs[1,1].axes.yaxis.set_visible(False)
        axs[1,1].set_title('Place Affordance')
        
        # Get action predictions
        l = str(info['lang_goal'])
        act = agent.act(img, info, goal=None)
        pick, place = act['pick'], act['place']
        
        # Visualize pick affordance
        pick_inp = {'inp_img': img, 'lang_goal': l}
        pick_conf = agent.attn_forward(pick_inp)
        logits = pick_conf.detach().cpu().numpy()

        pick_conf = pick_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0 = argmax[:2]
        p0_theta = (argmax[2] * (2 * np.pi / pick_conf.shape[2])) * -1.0
    
        line_len = 30
        pick0 = (pick[0] + line_len/2.0 * np.sin(p0_theta), pick[1] + line_len/2.0 * np.cos(p0_theta))
        pick1 = (pick[0] - line_len/2.0 * np.sin(p0_theta), pick[1] - line_len/2.0 * np.cos(p0_theta))

        if draw_grasp_lines:
            axs[1,0].plot((pick1[0], pick0[0]), (pick1[1], pick0[1]), color='r', linewidth=1)
        
        # Visualize place affordance
        # l = ['move the cyan ring to the middle of the stand' , 'move the yellow ring to the middle of the stand', 'move the gray ring to the middle of the stand']
        place_inp = {'inp_img': img, 'p0': pick, 'lang_goal': l}
        place_conf = agent.trans_forward(place_inp)

        place_conf = place_conf.permute(1, 2, 0)
        place_conf = place_conf.detach().cpu().numpy()
        argmax = np.argmax(place_conf)
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        p1_pix = argmax[:2]
        p1_theta = (argmax[2] * (2 * np.pi / place_conf.shape[2]) + p0_theta) * -1.0
        
        line_len = 30
        place0 = (place[0] + line_len/2.0 * np.sin(p1_theta), place[1] + line_len/2.0 * np.cos(p1_theta))
        place1 = (place[0] - line_len/2.0 * np.sin(p1_theta), place[1] - line_len/2.0 * np.cos(p1_theta))

        if draw_grasp_lines:
            axs[1,1].plot((place1[0], place0[0]), (place1[1], place0[1]), color='g', linewidth=1)
        
        # Overlay affordances on RGB input
        pick_logits_disp = np.uint8(logits * 255 * affordance_heatmap_scale).transpose(1,0,2)
        place_logits_disp = np.uint8(np.sum(place_conf, axis=2)[:,:,None] * 255 * affordance_heatmap_scale).transpose(1,0,2)    

        pick_logits_disp_masked = np.ma.masked_where(pick_logits_disp < 0, pick_logits_disp)
        place_logits_disp_masked = np.ma.masked_where(place_logits_disp < 0, place_logits_disp)

        axs[1][0].imshow(pick_logits_disp_masked, alpha=0.75)
        axs[1][1].imshow(place_logits_disp_masked, cmap='viridis', alpha=0.75)
        
        print(f"Lang Goal: {str(info['lang_goal'])}")
        plt.savefig(f'/home/pjw971022/RealWorldLLM/save_viz/cliport_only/heatmap/{eval_task}_{step}.png')
        
        # Act with the predicted actions
        # import ipdb;ipdb.set_trace()
        print(f"Pose 0: {act['pose0']}  Pose 1: {act['pose1']}")
        obs, reward, done, info = env.step(act)
        step += 1
        
    if done:
        print("Done. Success.")
    else:
        print("Max steps reached. Task failed.")
