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
    def __init__(self, ):
        self.use_clip = False
        self.image_dir = Path(f'/home/pjw971022/RealWorldLLM/save_viz/obs')

        # Set up ViLD object detector
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.score_threshold  = 0.1
        self.detector = OWLViTDetector(self.device, self.score_threshold)
        self.orcacle_receptacle = True
        self.bottom_z = 0.87
        self.recep_pose_dict = {    
                                'first paper': (0.65, -0.2, 0.15, 0, np.pi, 0),
                                'second paper': (0.65, -0.05, 0.15, 0, np.pi, 0),
                                'third paper': (0.65, 0.1, 0.15, 0, np.pi, 0),
                                'fourth paper': (0.65, 0.25, 0.15, 0, np.pi, 0),

                                'gray box': (0.6, 0.5, 0.2, 0, np.pi, 0),
                                'green box': (0.3, 0.5, 0.2, 0, np.pi, 0),
                                'trash can': (0.6, 0.5, 0.2, 0, np.pi, 0),
                                'box': (0.3, 0.5, 0.2, 0, np.pi, 0),
                                }

    def pointcloud_to_xyz(self, box, pointcloud):
        filtered_pc = self.extract_points_within_bbox(box, pointcloud[0])
        # mask =  filtered_pc[:,-1] > 0 
        # min_idx = filtered_pc[:,-1][mask].argmin() 
        # selected_point = filtered_pc[min_idx]

        filtered_pc = filtered_pc[filtered_pc[:, -1] > 0]
        sorted_array_by_z = filtered_pc[filtered_pc[:, -1].argsort()]  # Z 값 기준으로 정렬
        bottom_10_by_z = sorted_array_by_z[:10]  # Z 값이 가장 작은 하위 10개 선택
        selected_point = np.mean(bottom_10_by_z, axis=0)

        pose_x , pose_y = self.transform_coordinates(selected_point[0], selected_point[1])
        
        pose_z = self.bottom_z - selected_point[2]

        return pose_x, pose_y, pose_z

    def box_to_pose(self, box, pointcloud):
        pose_x, pose_y, pose_z = self.pointcloud_to_xyz(box, pointcloud)
        # pose_x, pose_y = self.transform_coordinates(box[0], box[1])  
        pose_z = 0.06 # 임시 세팅

        print(f"pose_x: {pose_x}  pose_y: {pose_y} pose_z: {pose_z}")
        assert (pose_x > MIN_MAX_X[0]) and (pose_x < MIN_MAX_X[1])
        assert (pose_y > MIN_MAX_Y[0]) and (pose_y < MIN_MAX_Y[1])
        # assert (pose_z > MIN_MAX_Z[0]) and (pose_z < MIN_MAX_Z[1])
        assert (pose_x**2 + pose_y**2) < MIN_MAX_DISTANCE
        
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
        pick_obj, place_obj = self.parse_action(data['lang_action'])

        text_queries = data['pick_objects']

        outputs = self.detector.forward(
            data['color'][0], text_queries,)
        detected_image_path = str(self.image_dir / f'detected_pick_obj.png')

        image_size = self.detector.model.config.vision_config.image_size
        image = cv.resize(data['color'][0].astype('float32'), (image_size, image_size))

        input_image = np.asarray(image) / 255.0
        self.detector.plot_predictions(input_image, text_queries, outputs, detected_image_path)

        # Select pick obj
        scores = outputs['scores']
        boxes = outputs['boxes']
        labels = outputs['labels']
        selected_box = None
        best_score = self.score_threshold
        for score, box, label in zip(scores, boxes, labels): # @ 개선의 여지 있음
            if (pick_obj == text_queries[label]) and (score > best_score):
                selected_box = box
                best_score = score
        
        # Calculate pick pose
        if selected_box is None:
            print("@@@ No Action")
            return -1
        else:
            pick_pose = self.box_to_pose(selected_box, data['pointcloud']) 
            print(f"Score: {best_score}")
            # Calculate place pose
            if self.orcacle_receptacle:
                place_pose = self.oracle_place_pose(place_obj)
            else:
                output = self.detector.forward(
                data['color'][0], [place_obj],)
                detected_place_image_path = str(self.image_dir / f'detected_place_obj.jpg')
                self.detector.plot_predictions(data['color'][0], [place_obj], output, detected_place_image_path)

            return {'pose0': pick_pose, 'pose1': place_pose}
        
    def parse_action(self, lang_action):
        """ parse action to retrieve pickup object and place object"""
        lang_action = re.sub(r'[^\w\s]', '', lang_action)  # remove all strings
        if self.task == 'real-world-making-word':
            target_pattern = r'\b\w+\s[A-Z]\b'
            recep_pattern = r"(firts paper|second paper|third paper)"
        else:
            raise NotImplementedError
        target_match = re.search(target_pattern, lang_action)
        recep_match = re.search(recep_pattern, lang_action)  # receptacle

        if target_match and recep_match:
            target = target_match.group(1)
            recep = recep_match.group(1)
            return target, recep
        else:
            return None, None

    def extract_points_within_bbox(self, bbox, point_cloud):
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