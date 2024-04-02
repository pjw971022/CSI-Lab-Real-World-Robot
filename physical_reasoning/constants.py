import os

SETTINGS_YAML_PATH = os.path.join(os.path.dirname(__file__), "latest_sembot_settings.yml")

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "raw_data/")

OVERALL_RAW_DATA_FILENAME = "epic-k_val_obj-list_data.json"
PATH_TO_OVERALL_RAW_DATA = os.path.join(RAW_DATA_DIR, OVERALL_RAW_DATA_FILENAME)
RAW_DATA_BY_EP_DIR = os.path.join(RAW_DATA_DIR, "raw_data_by_episode/")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output/")

ARE_VAL_DATA = True

EARLY_TERMINATION_TAG = "<early_termination>"
DONE_TAG = "<done>"
# JSONS_IN_RAW_DATA_FOLDER = [pos_json for pos_json in os.listdir(RAW_DATA_DIR) if pos_json.endswith('.json')]

# RAW_DATA_METADATA_JSON_PATH = os.path.join(os.path.dirname(__file__), "raw_data/metadata_json.json")  # outdated