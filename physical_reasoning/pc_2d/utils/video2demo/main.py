import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
import json
import argparse
from tqdm import tqdm
import numpy as np

from constants import SETTINGS_YAML_PATH, PATH_TO_OVERALL_RAW_DATA, RAW_DATA_BY_EP_DIR, OUTPUT_DIR, ARE_VAL_DATA
from utils import safe_mkdir
from gpt_reasoner import GPT_AS_REASONER
from physical_reasoning.physical_reasoner import SPATIAL_AS_REASONER

# Configure YAML to dump pretty multiline strings (https://github.com/yaml/pyyaml/issues/240#issuecomment-1096224358)
import yaml

def str_presenter(dumper, data):
    """configures yaml for dumping multiline strings
    Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data"""
    if data.count('\n') > 0:  # check for multiline string
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)
yaml.add_representer(str, str_presenter)
yaml.representer.SafeRepresenter.add_representer(str, str_presenter)

def make_all_dir(run_version):
    output_folder_path = os.path.join(OUTPUT_DIR, run_version)
    safe_mkdir(output_folder_path)
    return output_folder_path

def generate_prompt(prompt_data):
    sys_msg = prompt_data["system"]

    if "domain" in prompt_data:
        # replace domain
        sys_msg = sys_msg.replace("<domain>", prompt_data["domain"])

    prompt_dict = {
        "sys_msg": sys_msg,
    }

    return prompt_dict

def generate_raw_data_for_each_ep():
    with open(PATH_TO_OVERALL_RAW_DATA, "r") as fin:
        all_raw_data_list = json.load(fin)

    available_id_list = []
    ep_id, _, _= all_raw_data_list[0]["ep_id"].partition("__")
    ep_data = {ep_id: {}}
    for ep in all_raw_data_list:
        curr_ep_id, _, _= ep["ep_id"].partition("__")

        # assuming that we are seeing a new episode
        if ep_id != curr_ep_id:
            # save the old ep data for the previous ep_id
            ep_raw_data_path = os.path.join(RAW_DATA_BY_EP_DIR, f"{ep_id}_data.json")

            if not os.path.exists(ep_raw_data_path):
                # save the data
                with open(ep_raw_data_path, "w") as fout:
                    fout.write(json.dumps(ep_data))

            # move onto the next episode
            available_id_list.append(ep_id)
            ep_id = curr_ep_id
            ep_data = {ep_id: {}}
        
        tar_path = ep["tar_path"]
        if ARE_VAL_DATA:
            tar_path = tar_path.replace("/rgb_frames/", "/rgb_frames/")

        ep_data[ep_id][ep["id"]] = {
                "tar_path": tar_path,
                "start_image": ep["start_image"],
                "random_image": ep["image"],
                "end_image": ep["end_image"],
                "action": ep["action"],
                "states":[],
                "objects": ep["objects"]
            }

    # save the last video
    ep_raw_data_path = os.path.join(RAW_DATA_BY_EP_DIR, f"{ep_id}_data.json")

    if not os.path.exists(ep_raw_data_path):
        # save the data
        with open(ep_raw_data_path, "w") as fout:
            fout.write(json.dumps(ep_data))

    available_id_list.append(ep_id)
    
    return available_id_list


def generate_response_from_id_list(id_list, settings_dict, output_folder_path, full_restart):
    # from https://stackoverflow.com/questions/37506645/can-i-add-message-to-the-tqdm-progressbar
    for ep_id in (pbar := tqdm(id_list)):
        pbar.set_description(f"Video: {ep_id}")
        reasoner = GPT_AS_REASONER(ep_id, settings_dict, output_folder_path, full_restart)

        reasoner.generate()

        del reasoner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_restart", action="store_true", 
                        help="if included, it will overwrite gpt's existing predictions (not recommended because you lose all progress)")
    args = parser.parse_args()

    """Read prompt information from YAML"""
    print("Reading from YAML file...")
    with open(SETTINGS_YAML_PATH, "r") as f:
        settings_dict = yaml.safe_load(f)

    """save a copy of the yaml file"""
    yaml_path = os.path.join(os.path.dirname(__file__), "yaml_archive", f"{settings_dict['prompt_name']}_{settings_dict['yaml_version']}.yaml")
    with open(yaml_path, "w") as fout:
        yaml.dump(settings_dict, fout)

    """process the raw data into a usable dictionary"""
    # if that file exists, no need to regenerate
    all_available_ep_ids = generate_raw_data_for_each_ep()
    # print(f"available episodes and their ids (count = {len(all_available_ep_ids)}): {all_available_ep_ids}")

    ep_id_to_generate = settings_dict["generate_from_id"]
    if settings_dict["generate_all"]:
        if input("Warning: you are about to generate result for all videos. Are you sure (y/n)? ") != "y":
            sys.exit()
        else:
            ep_id_to_generate = all_available_ep_ids

    print(f"ep to gen: {ep_id_to_generate}")

    """create all the necessary folders"""
    run_version = f'{settings_dict["prompt_name"]}_{settings_dict["yaml_version"]}'
    output_folder_path = make_all_dir(run_version)

    """generate state-action for each video in the ep_id_to_generate list"""
    generate_response_from_id_list(ep_id_to_generate, settings_dict, output_folder_path, args.full_restart)
