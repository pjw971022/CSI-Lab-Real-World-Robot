import socket
import numpy as np
import cv2

def oracle_msg():
    return ((0.7, 0.3, 0.1, 0, np.pi, 0), (0.4, 0.0, 0.3, 0, np.pi, 0))

import zmq
import json
from PIL import Image
import io

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)  # EQ (REQUEST) 소켓
    socket.connect("tcp://115.145.175.206:5555")
    print("Client Start")
    # msg = oracle_msg()
    # msg = 'require human-cam image'
    # msg_json = json.dumps(msg)
    # socket.send_string(msg_json)

    # video_data = socket.recv()
    # with open('received_video.mp4', 'wb') as video_file:
    #     video_file.write(video_data)
    msg = 'require human-cam image'
    msg_json = json.dumps(msg)
    socket.send_string(msg_json)

    image_data = socket.recv()
    with open('received_image.png', 'wb') as image_file:
        image_file.write(image_data)
if __name__ == "__main__":
    main()
