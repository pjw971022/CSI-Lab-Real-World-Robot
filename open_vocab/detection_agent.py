# Author: Jimmy Wu
# Date: February 2023


from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.append('/home/pjw971022/RealWorldLLM/open_vocab')        
from owl_vit import OWLViTDetector
MIN_MAX_X = (0.2, 0.8)
MIN_MAX_Y = (-0.3, 0.6)
MIN_MAX_Z = (0.05, 0.3)

MIN_MAX_DISTANCE = 0.61
# (x^2 + y^2) < 0.61
# gray basket (0.6, 0,5)
# green basket (0.3, 0.5)
import re
class ObjectDetectorAgent:
    def __init__(self, task, pick_obj=None, grip_top=10):
        self.use_clip = False
        self.image_dir = Path(f'/home/pjw971022/RealWorldLLM/save_viz/obs')
        self.task = task
        self.grip_top = grip_top
        self.pick_obj = pick_obj
        # Set up ViLD object detector
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.score_threshold  = 0.1
        self.detector = OWLViTDetector(self.device, self.score_threshold)
        self.orcacle_receptacle = False
        self.bottom_z = 0.83 # @
        self.recep_pose_dict = { 
                                'anywhere':(0.55, -0.2, 0.15, 0, np.pi, 0),
                                'first paper': (0.65, -0.2, 0.15, 0, np.pi, 0),
                                'second paper': (0.65, -0.05, 0.15, 0, np.pi, 0),
                                'third paper': (0.65, 0.1, 0.15, 0, np.pi, 0),
                                'fourth paper': (0.65, 0.25, 0.15, 0, np.pi, 0),

                                'gray box': (0.6, 0.5, 0.3, 0, np.pi, 0),
                                'green box': (0.3, 0.5, 0.3, 0, np.pi, 0),
                                'trash can': (0.6, 0.5, 0.3, 0, np.pi, 0),
                                'box': (0.3, 0.5, 0.2, 0, np.pi, 0),
                                }

    def pointcloud_to_xyz(self, bbox, pointcloud, params):

        if params is None:
            filtered_pc = self.extract_points(bbox, pointcloud[0])
        elif params[0] == 'edge':
            print('param1: edge')
            filtered_pc = self.extract_edge_points(bbox, pointcloud[0])
        elif params[0] == 'center':
            print('param1: center')
            filtered_pc = self.extract_center_points(bbox, pointcloud[0])
        
        # mask =  filtered_pc[:,-1] > 0 
        # min_idx = filtered_pc[:,-1][mask].argmin() 
        # selected_point = filtered_pc[min_idx]

        filtered_pc = filtered_pc[filtered_pc[:, -1] > 0]
        sorted_array_by_z = filtered_pc[filtered_pc[:, -1].argsort()] 
        bottom_n_by_z = sorted_array_by_z[:self.grip_top] 
        selected_point = np.mean(bottom_n_by_z, axis=0)
  
        pose_x , pose_y = self.transform_coordinates(selected_point[0], selected_point[1])
        pose_z = self.bottom_z - selected_point[2]

        return pose_x, pose_y, pose_z

    def bbox_to_pose(self, bbox, pointcloud, params):
        pose_x, pose_y, pose_z = self.pointcloud_to_xyz(bbox, pointcloud, params)
        # pose_z = 0.06 # 임시 세팅

        print(f"pose_x: {pose_x}  pose_y: {pose_y} pose_z: {pose_z}")
        assert (pose_x > MIN_MAX_X[0]) and (pose_x < MIN_MAX_X[1])
        assert (pose_y > MIN_MAX_Y[0]) and (pose_y < MIN_MAX_Y[1])
        # assert (pose_z > MIN_MAX_Z[0]) and (pose_z < MIN_MAX_Z[1])
        assert (pose_x**2 + pose_y**2) < MIN_MAX_DISTANCE
        if params[1] == 'vertical':
            return (pose_x, pose_y, pose_z, np.pi/2, np.pi, 0)
        else:
            return (pose_x, pose_y, pose_z, 0, np.pi, 0)
        

    def transform_coordinates(self, x, y): # @
        # transform pixel pose with robot coordinate
        new_x = y + 0.526
        new_y = x - 0.027
        return new_x, new_y

    def oracle_place_pose(self, receptacle):
        pose = self.recep_pose_dict[receptacle]
        return pose
    
    def forward(self, data: dict):
        # Object detection
        pick_obj, place_obj, params = self.parse_action(data['lang_action'])

        pick_queries = data['pick_objects']

        pick_outputs = self.detector.forward(
            data['color'][0], pick_queries,)
        detected_image_path = str(self.image_dir / f'detected_pick_obj.png')

        image_size = self.detector.model.config.vision_config.image_size
        image = cv.resize(data['color'][0].astype('float32'), (image_size, image_size))

        input_image = np.asarray(image) / 255.0
        self.detector.plot_predictions(input_image, pick_queries, pick_outputs, detected_image_path)

        # Select pick obj
        scores = pick_outputs['scores']
        boxes = pick_outputs['boxes']
        labels = pick_outputs['labels']
        selected_box = None
        best_score = self.score_threshold
        for score, box, label in zip(scores, boxes, labels): # @ 개선의 여지 있음
            if (pick_obj == pick_queries[label]) and (score > best_score):
                selected_box = box
                best_score = score
        
        # Calculate pick pose
        if selected_box is None:
            print("@@@ No Action")
            return -1
        else:
            pick_pose = self.bbox_to_pose(selected_box, data['pointcloud'], params) 
            print(f"Score: {best_score}")
            # Calculate place pose
            if self.orcacle_receptacle:
                place_pose = self.oracle_place_pose(place_obj)
            else:
                place_outputs = self.detector.forward(
                data['color'][0], [place_obj],)
                detected_place_image_path = str(self.image_dir / f'detected_place_obj.jpg')
                if place_obj in self.recep_pose_dict.keys():
                    place_pose = self.oracle_place_pose(place_obj)
                else:
                    scores = place_outputs['scores']
                    boxes = place_outputs['boxes']
                    labels = place_outputs['labels']
                    selected_box = None
                    best_score = self.score_threshold
                    for score, box, label in zip(scores, boxes, labels):
                        if (pick_obj == pick_queries[label]) and (score > best_score):
                            selected_box = box
                            best_score = score
                    
                    self.detector.plot_predictions(data['color'][0], [place_obj], place_outputs, detected_place_image_path)
                    place_pose = self.bbox_to_pose(selected_box, data['pointcloud'], params) 

            return {'pose0': pick_pose, 'pose1': place_pose}
        
    def parse_action(self, lang_action):
        """ parse action to retrieve pickup object and place object"""
        lang_action = re.sub(r'[^\w\s]', '', lang_action)  # remove all strings
        if self.task == 'real-world-voice2demo':
            target_pattern = r'\b\w+\s[A-Z]\b'
            if 'move' in lang_action:
                recep_pattern = r"(red cube|yellow cube|green cube|green basket|gray basket|anywhere|near)"
            else:
                recep_pattern = r"()"
            # elif 'open' in lang_action:
            #     recep_pattern = r"()"
            # elif 'wipe' in lang_action:
            #     recep_pattern = r"()"
            # elif 'push' in lang_action:
            #     recep_pattern = r"()"

        elif self.task == 'real-world-making-word':
            target_pattern = r'\b\w+\s[A-Z]\b'
            recep_pattern = r"(firts paper|second paper|third paper)"
        elif self.task == 'debug':
            target_pattern = rf"({self.pick_obj})"
            recep_pattern = r"(green box|gray box)"
            param1_pattern = r"(edge|center)"
            param2_pattern = r"(vertical|horizontal)"
        else:
            raise NotImplementedError
        target_match = re.search(target_pattern, lang_action)
        recep_match = re.search(recep_pattern, lang_action)  # receptacle
        if self.task == 'debug':
            param1_match = re.search(param1_pattern, lang_action) 
            param2_match = re.search(param2_pattern, lang_action) 

        if target_match and recep_match:
            target = target_match.group(1)
            recep = recep_match.group(1)
            if self.task == 'debug':
                param1 = param1_match.group(1)
                param2 = param2_match.group(1)
                return target, recep, [param1,param2]
            return target, recep, None
        
        else:
            return None, None, None

    def extract_points(self, bbox, point_cloud):
        cx, cy, w, h = bbox
        cx *= 640
        cy *= 480
        w *= 640
        h *= 480
        rows, cols, _ = point_cloud.shape

        # Calculate the bounds of the bounding box
        x_min = max(int(cx - w / 2), 0)
        x_max = min(int(cx + w / 2), cols)
        y_min = max(int(cy - h / 2), 0)
        y_max = min(int(cy + h / 2), rows)

        # Create a mask for points within the bounding box
        y_indices, x_indices = np.ogrid[:rows, :cols]
        mask = (x_indices >= x_min) & (x_indices < x_max) & (y_indices >= y_min) & (y_indices < y_max)

        # Filter the point cloud based on the mask
        filtered_points = point_cloud[mask]
        return filtered_points
    
    def extract_edge_points(self, bbox, point_cloud, edge_ratio=0.1):
        cx, cy, w, h = bbox
        cx *= 640
        cy *= 480
        w *= 640
        h *= 480
        rows, cols, _ = point_cloud.shape

        # Calculate the bounds of the inner bounding box based on edge_ratio
        inner_w = w * (1 - edge_ratio * 2)
        inner_h = h * (1 - edge_ratio * 2)
        inner_x_min = max(int(cx - inner_w / 2), 0)
        inner_x_max = min(int(cx + inner_w / 2), cols)
        inner_y_min = max(int(cy - inner_h / 2), 0)
        inner_y_max = min(int(cy + inner_h / 2), rows)

        # Calculate the outer bounds of the bounding box
        x_min = max(int(cx - w / 2), 0)
        x_max = min(int(cx + w / 2), cols)
        y_min = max(int(cy - h / 2), 0)
        y_max = min(int(cy + h / 2), rows)

        # Create a mask for points at the edge of the bounding box
        y_indices, x_indices = np.ogrid[:rows, :cols]
        mask_edge = ((x_indices < inner_x_min) | (x_indices >= inner_x_max) | 
                    (y_indices < inner_y_min) | (y_indices >= inner_y_max)) & \
                    ((x_indices >= x_min) & (x_indices < x_max) & 
                    (y_indices >= y_min) & (y_indices < y_max))

        # Filter the point cloud based on the mask
        edge_points = point_cloud[mask_edge]
        return edge_points
    
    def extract_center_points(self, bbox, point_cloud, center_area=0.5):
        cx, cy, w, h = bbox
        cx *= 640
        cy *= 480
        w *= 640
        h *= 480
        rows, cols, _ = point_cloud.shape

        # Calculate the bounds of the center of the bounding box
        center_x_min = max(int(cx - w * center_area / 2), 0)
        center_x_max = min(int(cx + w * center_area / 2), cols)
        center_y_min = max(int(cy - h * center_area / 2), 0)
        center_y_max = min(int(cy + h * center_area / 2), rows)

        # Create a mask for points in the center of the bounding box
        y_indices, x_indices = np.ogrid[:rows, :cols]
        mask_center = (x_indices >= center_x_min) & (x_indices < center_x_max) & \
                    (y_indices >= center_y_min) & (y_indices < center_y_max)

        # Filter the point cloud based on the mask
        center_points = point_cloud[mask_center]
        return center_points