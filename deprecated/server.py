import socket
import json
import numpy as np
import pyrealsense2 as rs
from pick_and_place_module.pick_and_place import PickAndPlace, PrimitiveSkill
# import speech_recognition as sr

#############################################################################################
#### Desk ####
# Joint Unlock
# Activate FCI
#### In Server computer ####
# roscore
# roslaunch panda_moveit_config franka_control.launch load_gripper:=true robot_ip:=172.16.0.2
#############################################################################################

import tf.transformations as tf_trans

primitive_skill = PrimitiveSkill(0.05, 0.4, push_depth=0.03) 

def primitive_skill_fn(pose0 = None, pose1 = None, mode: int = 0, gripper_force: int = 30):
    if mode != 8:
        target_x, target_y, target_z, target_roll, target_pitch, target_yaw = pose0
        primitive_skill.setTargetPose(target_x, target_y, target_z, target_roll, target_pitch, target_yaw)

    if mode == 0:
        pose0_x, pose0_y, pose0_z, pose0_roll, pose0_pitch, pose0_yaw = pose0
        pose1_x, pose1_y, pose1_z, pose1_roll, pose1_pitch, pose1_yaw = pose1
        primitive_skill.setPose0(pose0_x, pose0_y, pose0_z, pose0_roll, pose0_pitch, pose0_yaw)
        primitive_skill.setPose1(pose1_x, pose1_y, pose1_z, pose1_roll, pose1_pitch, pose1_yaw)
        primitive_skill.execute_pick_and_place(gripper_force=gripper_force)
    elif mode == 1:
        primitive_skill.execute_pick(gripper_force=gripper_force)    
    elif mode == 2:
        primitive_skill.execute_place()    
    elif mode == 3:
        primitive_skill.execute_push(gripper_force=gripper_force)    
    elif mode == 4:
        primitive_skill.execute_pull(gripper_force=gripper_force) # @
    elif mode == 5:
        primitive_skill.execute_sweep()
    elif mode == 6:
        primitive_skill.execute_rotate(gripper_force=gripper_force)
    elif mode == 7:
        primitive_skill.execute_go()
    elif mode == 8:
        primitive_skill.go_to_ready_pose()
    

import zmq
import json
import time
import cv2
def apply_depth_filters(aligned_depth_frame):
    # Decimation filter
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 2)

    # Spatial filter
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 1)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    spatial.set_option(rs.option.holes_fill, 3)

    # Temporal Filter
    temporal = rs.temporal_filter()

    # Disparity transformation
    disparity_to_depth = rs.disparity_transform(False)
    depth_to_disparity = rs.disparity_transform(True)

    # Apply filters
    filtered_depth = aligned_depth_frame
    filtered_depth = depth_to_disparity.process(filtered_depth)
    filtered_depth = decimation.process(filtered_depth)
    filtered_depth = spatial.process(filtered_depth)
    filtered_depth = temporal.process(filtered_depth)
    filtered_depth = disparity_to_depth.process(filtered_depth)
    return filtered_depth

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # REP (REPLY) socket
    socket.bind("tcp://*:5555")
    print("Franka Server Open")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    dev = pipeline.get_active_profile().get_device()
    depth_sensor = dev.first_depth_sensor()
    depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
    
    # Set exposure and gain for color sensor (if needed)
    # color_sensor = dev.query_sensors()[1]  # Assuming [1] is the RGB sensor.
    # color_sensor.set_option(rs.option.enable_auto_exposure, 0)
    # color_sensor.set_option(rs.option.exposure, 100)
    # color_sensor.set_option(rs.option.gain, 50)

    align_to = rs.stream.color 
    align = rs.align(align_to)
    while True:
        action_json = socket.recv_string()
        action_tuple = json.loads(action_json)
        print(f"Receive Action!   act: {action_tuple}")
        if action_tuple == 0: # reset
            pass
        elif action_tuple == 1: # server close
            break
        elif action_tuple == 2: # go to ready pose
            primitive_skill.go_to_ready_pose()
        elif isinstance(action_tuple[0], int):
            print(action_tuple)
            pick_pose = action_tuple[1]
            place_pose = action_tuple[2]
            if place_pose is not None:
                print("pick_pose: ", pick_pose,"place_pose: ", place_pose)
                primitive_skill_fn(tuple(pick_pose), tuple(place_pose), mode=action_tuple[0])
            elif pick_pose is not None:
                primitive_skill_fn(tuple(pick_pose), mode=action_tuple[0])
            else:
                primitive_skill_fn(mode=action_tuple[0])
        else:
            print("### Action Error ###")
            pass
            
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        # Apply depth filters
        aligned_depth_frame = apply_depth_filters(aligned_frames.get_depth_frame())
        # aligned_depth_frame = aligned_frames.get_depth_frame()  # Aligned depth frame is a depth frame
        color_frame = aligned_frames.get_color_frame()  # Aligned color frame is a regular color frame

        if not aligned_depth_frame or not color_frame:
            continue

        depth_array = np.asanyarray(aligned_depth_frame.get_data())
        rgb_array = np.asanyarray(color_frame.get_data())

        cv2.imwrite('/home/franka/fr3_workspace/franka_env/send_image.png', rgb_array)

        pc = rs.pointcloud()
        pc.map_to(aligned_depth_frame)
        points = pc.calculate(aligned_depth_frame)

        vtx = np.asanyarray(points.get_vertices())
        rgb_list = rgb_array.tolist()
        depth_list = depth_array.tolist()
        vtx_list = vtx.tolist()
        data = json.dumps({'rgb': rgb_list, 'depth': depth_list, 'pointcloud': vtx_list}) # 

        socket.send_string(data)
        print("Send Image!")
    print("Franka Server Close")


if __name__ == "__main__":
    main()