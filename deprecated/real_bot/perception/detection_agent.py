from pathlib import Path
import numpy as np
import torch
import sys
sys.path.append('/home/pjw971022/Sembot/real_bot/perception')        
from owl_vit import OWLViTDetector
MIN_MAX_X = (0.2, 0.8)
MIN_MAX_Y = (-0.4, 0.6)
MIN_MAX_Z = (0.03, 0.6)
MIN_MAX_DISTANCE = 0.61

import re
from PIL import Image
import cv2
from transformers.image_utils import ImageFeatureExtractionMixin
mixin = ImageFeatureExtractionMixin()

class ObjectDetectorAgent:
    def __init__(self, task, target1_obj=None, grip_top=500):
        self.use_clip = False
        self.image_dir = Path(f'/home/pjw971022/Sembot/real_bot/save_viz/obs')
        self.task = task
        self.grip_top = grip_top
        self.target1_obj = target1_obj
        # Set up ViLD object detector
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.score_threshold  = 0.2
        self.detector = OWLViTDetector(self.device, self.score_threshold)
        self.bottom_z = 0.865 # @
        self.recep_pose_dict = { 
                'anywhere':(0.5, -0.3, 0.15, 0, np.pi, 0),
                
                'first paper': (0.65, -0.35, 0.02, 0, np.pi, 0),
                'second paper': (0.65, -0.27, 0.02, 0, np.pi, 0),
                'third paper': (0.65, -0.19, 0.02, 0, np.pi, 0),
                'fourth paper': (0.65, -0.11, 0.02, 0, np.pi, 0),
                'fifth paper': (0.65, -0.03, 0.02, 0, np.pi, 0),
                'sixth paper': (0.65, 0.05, 0.02, 0, np.pi, 0),
                'seventh paper': (0.65, 0.13, 0.02, 0, np.pi, 0),
                'eighth paper': (0.65, 0.21, 0.02, 0, np.pi, 0),
                'nineth paper': (0.65, 0.29, 0.02, 0, np.pi, 0),
                'tenth paper': (0.65, 0.37, 0.02, 0, np.pi, 0),

                'gray basket': (0.4, 0.35, 0.2, 0, np.pi, 0),
                'green basket': (0.4, 0.34, 0.25, 0, np.pi, 0),
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

        filtered_pc = filtered_pc[filtered_pc[:, -1] > 0]
        sorted_array_by_z = filtered_pc[filtered_pc[:, -1].argsort()] 
        bottom_n_by_z = sorted_array_by_z[:self.grip_top] 
        selected_point = np.mean(bottom_n_by_z, axis=0)
  
        pose_x , pose_y = self.transform_coordinates(selected_point[0], selected_point[1])
        pose_z = self.bottom_z - selected_point[2]

        return pose_x, pose_y, pose_z

    def bbox_to_pose(self, bbox, pointcloud, params):
        pose_x, pose_y, pose_z = self.pointcloud_to_xyz(bbox, pointcloud, params)
        pose_z = 0.083

        print(f"pose_x: {pose_x}  pose_y: {pose_y} pose_z: {pose_z}")
        assert (pose_x > MIN_MAX_X[0]) and (pose_x < MIN_MAX_X[1])
        assert (pose_y > MIN_MAX_Y[0]) and (pose_y < MIN_MAX_Y[1])
        assert (pose_z > MIN_MAX_Z[0]) and (pose_z < MIN_MAX_Z[1])
        assert (pose_x**2 + pose_y**2) < MIN_MAX_DISTANCE

        return pose_x, pose_y, pose_z, 0, np.pi, 0
    
    def transform_coordinates(self, x, y): # @
        # transform pixel pose with robot coordinate
        new_x = y + 0.54
        new_y = x - 0.13
        return new_x, new_y

    def oracle_target_pose(self, receptacle):
        pose = self.recep_pose_dict[receptacle]
        return pose
    
    def forward(self, data: dict):
        # Object detection
        mode, target1_obj, target2_obj, params = self.parse_action(data['lang_action'])
        target1_queries = data['objects']
        target2_queries = data['objects']
        print(f"target1_obj: {target1_obj}   target2_obj: {target2_obj}")

        if 'ready' in target1_obj :
            pass
        elif 'anywhere' in target1_obj:
            pass
        else:
            assert target1_obj in target1_queries

        if 'anywhere' in target2_obj:
            pass
        elif target2_obj is not None:
            assert target2_obj in target2_queries

        rgb_path = '/home/pjw971022/Sembot/real_bot/save_viz/obs/image_obs.png'
        image = Image.open(rgb_path).convert("RGB")

        target1_outputs = self.detector.forward(image, target1_queries)
        target1_detected_image_path = str(self.image_dir / f'detected_target1_obj.png')

        image_size = self.detector.model.config.vision_config.image_size
        image = mixin.resize(image, image_size)
        input_image = np.asarray(image).astype(np.float32) / 255.0
        resized_image = cv2.resize(input_image, (640, 480))

        self.detector.plot_predictions(resized_image, target1_queries, target1_outputs, target1_detected_image_path)

        scores = target1_outputs['scores']
        boxes = target1_outputs['boxes']
        labels = target1_outputs['labels']
        selected_box = None
        best_score = self.score_threshold
        for score, box, label in zip(scores, boxes, labels): 
            if (target1_obj == target1_queries[label]) and (score > best_score):
                selected_box = box
                best_score = score
        
        if 'ready' in target1_obj:
            return  {'mode': mode, 'pose0': None, 'pose1': None}
        elif 'anywhere' in target1_obj:
            target1_pose = self.oracle_target_pose(target1_obj)
            print("anywher: ", target1_pose)
            noise = np.random.normal(loc=0, scale=0.2, size=2)
            target1_pose[0] += noise[0]
            target1_pose[0] += noise[1]
            
            return  {'mode': mode, 'pose0': target1_pose, 'pose1': None}
        elif selected_box is None:
            print("@@@ No Action")
            return -1
        else:
            target1_pose = self.bbox_to_pose(selected_box, data['pointcloud'], params) 
            if target2_obj is not None:
                target2_outputs = self.detector.forward(image, target2_queries,)
                target2_detected_image_path = str(self.image_dir / f'detected_target2_obj.jpg')
                if target2_obj in self.recep_pose_dict.keys():
                    target2_pose = self.oracle_target_pose(target2_obj)
                else:
                    scores = target2_outputs['scores']
                    boxes = target2_outputs['boxes']
                    labels = target2_outputs['labels']
                    selected_box = None
                    best_score = self.score_threshold
                    for score, box, label in zip(scores, boxes, labels):
                        if (target2_obj == target2_queries[label]) and (score > best_score):
                            selected_box = box
                            best_score = score
                    self.detector.plot_predictions(resized_image, target2_queries, target2_outputs, target2_detected_image_path)
                    target2_pose = self.bbox_to_pose(selected_box, data['pointcloud'], params) 
        
                    print(f"target2 pose: {target2_pose}")
            else:
                target2_pose = None

            return {'mode':mode, 'pose0': target1_pose, 'pose1': target2_pose}
        
    def parse_action(self, lang_action):
        """ parse action to retrieve target1up object and target2 object"""
        # lang_action = re.sub(r'[^\w\s]', '', lang_action)  # remove all strings
        if 'move' in lang_action:
            mode = 0
        elif 'pick' in lang_action:
            mode = 1
        elif 'place' in lang_action:
            mode = 2
        elif 'push' in lang_action:
            mode = 3
        elif 'pull' in lang_action:
            mode = 4
        elif 'sweep' in lang_action:
            mode = 5
        elif 'rotate' in lang_action:
            mode = 6
        elif 'ready' in lang_action:
            mode = 8
        elif 'go' in lang_action:
            mode = 7
        else:
            raise NotImplementedError
                
        if self.task == 'speech2demo':
            target_pattern = "\<(.*?)\>"
            if 'move' in lang_action:
                if 'letter' in lang_action:
                    recep_pattern = r"(first paper|second paper|third paper|fourth paper|fifth paper|sixth paper|seventh paper|eighth paper|nineth paper|tenth paper)"
                else:
                    recep_pattern = r"(red block|yellow block|green block|green basket|gray basket|napkin|anywhere)"
            else:
                recep_pattern = r"()"

        elif self.task == 'real-world-making-word':
            target_pattern = r'\b\w+\s[A-Z]\b'
            recep_pattern = r"(firts paper|second paper|third paper)"
        elif self.task == 'debug':
            target_pattern = rf"({self.target1_obj})"
            recep_pattern = r"(green box|gray box)"
            param1_pattern = r"(edge|center)"
            param2_pattern = r"(vertical|horizontal)"
        else:
            raise NotImplementedError
        target_matchs = re.findall(target_pattern, lang_action)
        recep_match = re.search(recep_pattern, lang_action)  # receptacle
        if self.task == 'debug':
            param1_match = re.search(param1_pattern, lang_action) 
            param2_match = re.search(param2_pattern, lang_action) 

        if target_matchs and recep_match:
            if mode == 0:
                target = target_matchs[0]
                recep = target_matchs[1]
            else:
                target = target_matchs[0]
                recep = None

            if self.task == 'debug':
                param1 = param1_match.group(1)
                param2 = param2_match.group(1)
                return target, recep, [param1, param2]
            return mode, target, recep, None
        
        else:
            return None, None, None, None

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