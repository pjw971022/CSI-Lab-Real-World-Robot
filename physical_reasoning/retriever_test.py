import vertexai
import os
WORKSPACE = "/home/pjw971022/workspace"
video_folder_path = WORKSPACE + '/Sembot/physical_reasoning/database/CharadesEgo/CharadesEgo_v1_480'
video_description_prompt = "Please explain in detail, focusing on the actions of the arms and hands of the person in the video. When explaining, you must pair primitive actions such as move, grasp, and rotate with the context of the action. Example: <move to the cabinet> - To open a cabinet with a handle, move your hand towards the cabinet."
PROJECT_ID = "gemini-api-415903"  # @param {type:"string"}
LOCATION = "asia-northeast3"  # @param {type:"string"}
key_path = WORKSPACE + "Sembot/physical_reasoning/gemini-api-415903-0f8224218c2c.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

vertexai.init(project=PROJECT_ID, location=LOCATION)

from retrieval_utils import VideoRetriever
retriever = VideoRetriever(video_folder_path,video_description_prompt)