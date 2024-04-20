import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from moviepy.editor import VideoFileClip
from video_utils import upload_blob
import os

WORKSPACE = "/home/pjw971022/workspace/"
video_folder_path = '/home/mnt/data/video_rag_database/CharadesEgo/CharadesEgo_v1_480'
output_folder = "/home/pjw971022/workspace/Sembot/physical_reasoning/output_video"

# 1. gcloud CLI 설치: https://cloud.google.com/sdk/docs/install?hl=ko#deb
# 2. gcloud login https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-multimodal-embeddings?hl=ko#python

#--------------------------------------------------------------------------------

#############################################################
#############################################################

QUERY = "press the circular button"
video_description_prompt = '''Please explain in detail, focusing on the actions of the arms and hands of the person in the video.
When explaining, you must pair primitive actions such as move, grasp, and rotate with the context of the action.
Example: <move to the cabinet> - To open a cabinet with a handle, move your hand towards the cabinet.'''

#############################################################
#############################################################

#--------------------------------------------------------------------------------

from video_utils import VideoRetriever
metadata_path = '/home/mnt/data/video_rag_database/embedding_data/CharadesEgo_emb_only_metadata.csv'
retriever = VideoRetriever(video_folder_path,
                           video_description_prompt,
                           use_video_description=False,
                           metadata_path=metadata_path)

# retrieved_video = retriever.find_video(QUERY)
# retriever.save_multiple_video_segments(retrieved_video, output_folder)

#--------------------------------------------------------------------------------

vision_config = {"temperature": 0.2}
key_path = WORKSPACE + "Sembot/physical_reasoning/video_rag/gemini-video-0403-0476b10bf020.json"
PROJECT_ID = "gemini-video-0403"
LOCATION = "asia-northeast3"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
vertexai.init(project=PROJECT_ID, location=LOCATION)
multimodal_model = GenerativeModel("gemini-1.0-pro-vision")


#--------------------------------------------------------------------------------

result_context = []


video_uri = upload_blob(f"{output_folder}/top1.mp4", "top1.mp4")
video_file = Part.from_uri(video_uri, mime_type="video/mp4")
contents = [video_file, video_description_prompt]
response = multimodal_model.generate_content(contents, generation_config=vision_config)
description_video = response.text
print(description_video)