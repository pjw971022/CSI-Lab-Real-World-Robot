
import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from google.cloud import storage
import openai
openai.api_key = "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"
from openai import OpenAI
WORKSPACE = "/home/pjw971022/workspace"
location = "asia-northeast3"
project_id = "gemini-api-415903"
key_path = WORKSPACE + "/Sembot/physical_reasoning/gemini-api-415903-0f8224218c2c.json"

#-----------------------------------------------------------------------------


prompt_demo_extractor ="""
Please explain in detail, focusing on the actions of the arms and hands of the person in the video. When explaining, you must pair primitive actions such as move, grasp, and rotate with the context of the action. Example: <move to the cabinet> - To open a cabinet with a handle, move your hand towards the cabinet."""

#-----------------------------------------------------------------------------

def upload_blob(source_file_name, destination_blob_name, bucket_name = 'expert_video_demo'):
    storage_client = storage.Client(project='gemini-api-413603')
    bucket = storage_client.bucket(bucket_name)
    
    # 파일 업로드
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    
    # 파일에 공개 액세스 권한 설정
    blob.make_public()
    print(f"###### upload blob: {destination_blob_name}")
    # 파일의 URI 반환
    return f"gs://{bucket_name}/{destination_blob_name}"


#-----------------------------------------------------------------------------
class VideoLLMQuery:
    def __init__(self, video_folder_path, video_description_prompt) -> None:
        self.vision_config = {"max_output_tokens": 800, "temperature": 0.2, "top_p": 1, "top_k": 32}
        self.text_config = {"max_output_tokens": 512, "temperature": 0.0, "top_p": 1}
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        self.model = self.setup(location, project_id)
        self.retriever = VideoRetriever(video_folder_path, video_description_prompt)

    def setup(self, location, project_id):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        vertexai.init(project=project_id, location=location)
        vision_model = GenerativeModel("gemini-pro-vision")
        return vision_model
    
    def gen_caption(self, video="demo_video.mp4"):
        video_uri = upload_blob(video, video)
        video_file = Part.from_uri(video_uri, mime_type="video/mp4")

        contents = [video_file, prompt_demo_extractor ]
        response_2 = self.model.generate_content(contents, generation_config=self.vision_config) #, safety_settings = self.safety_settings
        description_video = response_2.text
        return description_video
    
    
    def query_to_video(self, question, video_path="demo_video.mp4"):

        video_uri = upload_blob(video_path, video_path.split('/')[-1])
        video_file = Part.from_uri(video_uri, mime_type="video/mp4")

        contents = [video_file, question]
        response = self.model.generate_content(contents, generation_config=self.vision_config) #, safety_settings = self.safety_settings
        description_video = response.text
        return description_video

from video_rag.utils import get_similar_video_from_query, get_video_document_metadata
PROJECT_ID = "gemini-api-415903"  # @param {type:"string"}
LOCATION = "asia-northeast3"  # @param {type:"string"}
key_path = WORKSPACE + "/Sembot/physical_reasoning/gemini-api-415903-0f8224218c2c.json"

class VideoRetriever:
    def __init__(self, 
                 video_folder_path,
                 video_description_prompt,
                 use_video_description,
                 metadata_path):
        self.multimodal_model = self.setup()

        self.video_metadata_df = get_video_document_metadata(
            self.multimodal_model,  # we are passing gemini 1.0 pro vision model
            video_folder_path=video_folder_path,
            video_description_prompt=video_description_prompt,
            embedding_size=1408,
            metadata_path=metadata_path,
            use_video_description=use_video_description,
            # add_sleep_after_page = True, # Uncomment this if you are running into API quota issues
            # generation_config = # see next cell
            # safety_settings =  # see next cell
        )

    def setup(self,):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
        return multimodal_model
    
    def find_video(self, query, top_n=1):
        retrieved_video = get_similar_video_from_query(
            self.video_metadata_df,
            query,
            column_name="video_embedding_from_video_only",
            top_n=top_n,
            # chunk_text=True,
            embedding_size=1408
        )
        return retrieved_video
    