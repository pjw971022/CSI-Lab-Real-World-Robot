import hydra
from omegaconf import DictConfig, OmegaConf
import openai
from arguments import get_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from envs.rlbench_env import VoxPoserRLBench
from utils import set_lmp_objects,load_prompt
import numpy as np
from rlbench import tasks
import os
os.environ['OPENAI_API_KEY'] =  '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' 
import wandb
import numpy as np
import sys 
sys.path.append('/home/pjw971022/workspace/Sembot/physical_reasoning')
from physical_reasoning.interactive_agent import InteractiveAgent
MAX_TRY = 20
CUSTOM_TASKS = [
    #######################   LLM으로도 가능   ############################
    ######  무난하게 성공  ######
    tasks.PutRubbishInBin,
    # tasks.LampOff,
    # tasks.PutKnifeOnChoppingBoard,
    # tasks.PushButton,
    #  tasks.TakeUmbrellaOutOfUmbrellaStand, # umbrella haddle 
    # tasks.CloseLaptopLid, # llama3/gpt-4는 성공
    # tasks.SlideBlockToTarget,
    # tasks.PhoneOnBase,
    # tasks.OpenJar,

    #### 어거지이긴 하지만 성공 ####
    # tasks.TakePlateOffColoredDishRack,
    # tasks.TakeToiletRollOffStand,

    #######################################################
    ######   expert 이미 만듬 #####
    # tasks.TakeOffWeighingScales,
    #  tasks.PutGroceriesInCupboard,
    # tasks.TakeLidOffSaucepan,

    ######   해볼만한 테스크   ######   
    # tasks.TakeMoneyOutSafe
    # tasks.StraightenRope,
    # tasks.PutPlateInColoredDishRack
    # tasks.StackWine
    # tasks.StackChairs,
    # tasks.PressSwitch,
    # tasks.OpenWindow
    # tasks.OpenGrill
    # tasks.UnplugCharger # 수평으로 이동해서 빼야되는거 LLM이 몰름
    # tasks.OpenWineBottle, # 뚜껑이 잘 열리지 않음

    ######################### 방향 맞추기 어려움 #####################
    #  tasks.PutToiletRollOnStand, # 껴넣는 방향을 알아야 넣을 수 있음
    
    #  tasks.PutBooksOnBookshelf,
    #  tasks.PlugChargerInPowerSupply, # 방향 맞추기 매우 어려워 보임
    #  tasks.ChangeClock,# 방향 맞추기 매우 어려워 보임
    # tasks.TurnOvenOn,# 방향 맞추기 매우 어려워 보임


    #######################  Detection Error #####################
    #  tasks.MeatOffGrill, # grill lid / target position
    #  tasks.OpenMicrowave, # microwave handle 
    #  tasks.OpenWashingMachine, # washing machine handle 
    #  tasks.GetIceFromFridge, # ice dispenser 
    #  tasks.PutBottleInFridge, # fridge handle 
    #  tasks.InsertUsbInComputer, # usb port 
    #  tasks.WipeDesk, # 먼지 위치
    # tasks.SweepToDustpan, # broom stick 부분
    # tasks.BasketballInHoop, # hoop
    # tasks.WaterPlants # handle
    
    ####################### 어려워 보이는 테스크  ####################
    ######   Dynamic object   ######
    #  tasks.MoveHanger, 
]

from PIL import Image
import time
@hydra.main(config_path=f'./configs', config_name="rlbench_config")
# @profile
def main(cfgs: DictConfig):
    config = cfgs
    plain_dict = OmegaConf.to_container(config, resolve=True, enum_to_str=True)

    # uncomment this if you'd like to change the language model (e.g., for faster speed or lower cost)
    for lmp_name, cfg in config['lmp_config']['lmps'].items():
        cfg['model'] =  'llama3-70b-server' #'gemini-1.0-pro-latest' # # 'gpt-4-1106-preview' #' # # 

    # initialize env and voxposer ui
    use_server = config['use_server']
    use_sembot = config['use_sembot']
    use_oracle_plan = config['use_oracle_plan']
    use_oracle_instruction = config['use_oracle_instruction']
    use_human_agent = config['use_human_agent']
    save_pcd = config['save_pcd']

    image_types = ['front_rgb', 'left_shoulder_rgb', 'right_shoulder_rgb', 'overhead_rgb', 'wrist_rgb']

    visualizer = ValueMapVisualizer(config['visualizer']) # None
    env = VoxPoserRLBench(visualizer=visualizer,
                           save_pcd=save_pcd,
                              use_server=use_server,
                              server_ip=config['server_ip'],)
    lmps, lmp_env = setup_LMP(env, config, debug=False)
    voxposer_ui = lmps['plan_ui']
    composer_ui = lmps['composer_ui']

    interactive_agent = InteractiveAgent(composer_ui, config['interactive_agent'])
    task_list = CUSTOM_TASKS
    start_time = time.time()
    for task in task_list:
        env.load_task(task)
        task_name = env.task.get_name()
        wandb.init(project="voxposer-naive",
                   entity="pjw971022",
                   name=task_name, 
                   reinit=True)
        wandb.config.update(plain_dict)
        i = 0
        cnt = 0
        while i < cfgs['episode_length'] and cnt < MAX_TRY:
            # try:
            desc_, obs = env.reset()
            if len(desc_)==2:
                descriptions, target_obj = desc_
            else:
                descriptions = desc_
            set_lmp_objects(lmps, env.get_object_names())  # set the object names to be used by voxposer
            instruction = np.random.choice(descriptions) 
            env.instruction = instruction            
            wandb.log({"instruction": instruction})
            if use_human_agent:
                print("INSTRUCTION:", instruction)
                while True:
                    human_plan = input('INPUT:')
                    if human_plan == 'q':
                        break
                    else:
                        composer_ui(human_plan)
            elif use_oracle_plan:
                oracle_plan_code = load_prompt(f"{config['env_name']}/oracle_plan/{task_name}.txt")
                plan_list = oracle_plan_code.split('\n')
                data_dict = {'instruction': descriptions, 'plan':[]}
                for step, plan in enumerate(plan_list):
                    print("ORACLE PLAN:", plan)
                    obs = env.latest_obs
                    for type in image_types:
                        val = obs.get_attribute(type)
                        val = val.astype(np.uint8)
                        image = Image.fromarray(val)
                        image.save(f'/home/pjw971022/workspace/Sembot/sembot/src/visualizations/obs/{task_name}_{type}_{step}.png')
                    
                    data_dict['plan'].append(plan)
                    composer_ui(plan)
                    
            elif use_sembot:
                if use_oracle_instruction:
                    oracle_instruction = load_prompt(f"{config['env_name']}/oracle_instruction/{task_name}.txt")
                    obs_dict = {'instruction': oracle_instruction, 'possible_obj': env.get_object_names()}
                    interactive_agent(obs_dict) #
                else:
                    oracle_plan_code = None
                    obs_dict = {'instruction': instruction, 'possible_obj': env.get_object_names()}
                    instruction += f'. {interactive_agent.obs_captioning()}'
                    interactive_agent(obs_dict)
            else:
                if use_oracle_instruction:
                    oracle_instruction = load_prompt(f"{config['env_name']}/oracle_instruction/{task_name}.txt")
                    voxposer_ui(oracle_instruction) #
                else:      
                    voxposer_ui(instruction)
            i +=1
    print(f'*** Total episode time took {time.time() - start_time:.2f}s ***')
    wandb.finish()

if __name__ =='__main__':
    main()