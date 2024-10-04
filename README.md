## How to use Franka-Emika in Real-World

1. Connect with fr3 using ros in Franka PC
```bash
roscore
roslaunch panda_move_it franka_control.launch robot_ip:=172.16.0.2
```

2. Open the franka server in Franka PC
```bash
python3 /home/franka_env/server.py
```

3. Connect with franka server by example in Local PC
```bash
cd ravens/ravens/environments/
python3 client_example.py
```

4. Connect with franka server by gym
```bash
cd ravens/ravens/environments/
python3 environment_real.py
```

## How to make real-world dataset
1. Align the depth camera config with code in [cameras.RealSenseD435.CONFIG](ravens/ravens/environments/environment_real.py)
2. Generate real-world data by human labeling [`data_gen_real_world.sh`](scripts/data_gen_real_world.sh)
```bash
source data_gen_real_world.sh
```
3. Human labeling of reward/language with interface

## How to train real-world dataset
Run [`train_real.sh`](scripts/train_real.sh)
```bash
source train_real.sh
```

## How to test real-world dataset
### without LLM
Run [`eval.sh`](scripts/eval.sh)
```bash
source eval.sh
```
### with LLM
Run [`llm_eval_real.sh`](scripts/train_real.sh)
```bash
source llm_eval_real.sh
```
