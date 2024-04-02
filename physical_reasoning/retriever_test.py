import vertexai
import os

WORKSPACE = "/home/pjw971022/workspace"
video_folder_path = '/mnt/video_rag/database/CharadesEgo/CharadesEgo_v1_480'
video_description_prompt = "Please explain in detail, focusing on the actions of the arms and hands of the person in the video. When explaining, you must pair primitive actions such as move, grasp, and rotate with the context of the action. Example: <move to the cabinet> - To open a cabinet with a handle, move your hand towards the cabinet."


# 1. gcloud CLI 설치: https://cloud.google.com/sdk/docs/install?hl=ko#deb
# 2. gcloud login https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-multimodal-embeddings?hl=ko#python

from retrieval_utils import VideoRetriever
metadata_path = '/mnt/video_rag/database/embedding_data/CharadesEgo_emb_only_metadata.csv'
retriever = VideoRetriever(video_folder_path,
                           video_description_prompt,
                           use_video_description=False,
                           metadata_path=metadata_path)

# retriever.find_video("Open the bottom refrigerator door.")