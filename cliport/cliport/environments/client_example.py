import numpy as np


def oracle_action():
    return ((0.7, 0.3, 0.1, 0, np.pi, 0), (0.4, 0.0, 0.3, 0, np.pi, 0))

import zmq
import json
import cv2

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)  # REQ (REQUEST) 소켓
    socket.connect("tcp://115.145.175.206:5555")
    print("Client Start")
    action = 0 # oracle_action()

    action_json = json.dumps(action)
    socket.send_string(action_json)

    data = socket.recv_string()
    data = json.loads(data)

    rgb_array = np.array(data['rgb'])
    depth_array = np.array(data['depth'])
    pointcloud_array = np.array(data['pointcloud'])
    
    cv2.imwrite('/home/pjw971022/RealWorldLLM/save_viz/obs/image_obs.png', rgb_array)
    print("save image!")
if __name__ == "__main__":
    main()
