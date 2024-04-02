
from constants import SETTINGS_YAML_PATH, PATH_TO_OVERALL_RAW_DATA, RAW_DATA_BY_EP_DIR, OUTPUT_DIR, ARE_VAL_DATA
from physical_reasoning.physical_reasoner import Physical_AS_REASONER
import yaml

if __name__ == "__main__":
    """Read prompt information from YAML"""
    print("Reading from YAML file...")
    with open(SETTINGS_YAML_PATH, "r") as f:
        settings_dict = yaml.safe_load(f)
    reasoner = Physical_AS_REASONER(settings_dict)
    reasoner.generate_fg_skill_server()
    