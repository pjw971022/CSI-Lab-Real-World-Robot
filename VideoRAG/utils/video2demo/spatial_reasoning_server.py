import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
import json
import argparse
from tqdm import tqdm
import numpy as np

from constants import SETTINGS_YAML_PATH, PATH_TO_OVERALL_RAW_DATA, RAW_DATA_BY_EP_DIR, OUTPUT_DIR, ARE_VAL_DATA
from spatial_reasoner import SPATIAL_AS_REASONER
import yaml

def generate_response_from_id_list(settings_dict):
    reasoner = SPATIAL_AS_REASONER(settings_dict)
    reasoner.generate_fg_skill()


if __name__ == "__main__":
    """Read prompt information from YAML"""
    print("Reading from YAML file...")
    with open(SETTINGS_YAML_PATH, "r") as f:
        settings_dict = yaml.safe_load(f)

    generate_response_from_id_list(settings_dict)
