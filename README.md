# How to use Robocnam in Real-World

1. Connect with fr3 using ros
```bash
roscore
roslaunch panda_move_it franka_control.launch robot_ip:=172.16.0.2
```

2. Open the franka server
```bash
python3 /home/franka/fr3_workspace/franka_env/server.py
```

3. Connect with franka server
```bash
cd cliport/cliport/environments/
python3 client_example.py
```
