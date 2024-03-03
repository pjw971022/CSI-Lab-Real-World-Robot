cd /home/andykim0723/RLBench/
pip install .
cd /home/andykim0723/RLBench/Sembot/low_level_planner/src
python3 playground.py task_name=pour_from_cup_to_cup context_mode=pre_user_command
# python3 playground.py task_name=pour_from_cup_to_cup context_mode=vision_observation
# python3 playground.py task_name=pour_from_cup_to_cup context_mode=expert_demo

# put_rubbish_in_bin / put_groceries_in_cupboard / pour_from_cup_to_cup