
import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from google.cloud import storage
import openai
openai.api_key = "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"
from openai import OpenAI
WORKSPACE = "/home/pjw971022/workspace"
PROJECT_ID = "gemini-video-0403"  # @param {type:"string"}
LOCATION = "asia-northeast3"  # @param {type:"string"}
KEY_PATH = WORKSPACE + "/Sembot/physical_reasoning/video_rag/gemini-video-0403-0476b10bf020.json"


def upload_blob(source_file_name, destination_blob_name, bucket_name = 'expert_video_demo'):
    storage_client = storage.Client(project='gemini-video-0403')
    bucket = storage_client.bucket(bucket_name)
    
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    
    blob.make_public()
    print(f"###### upload blob: {destination_blob_name}")
    return f"gs://{bucket_name}/{destination_blob_name}"

class VideoLLMQuery:
    def __init__(self, model_dict) -> None:
        self.vision_config = {"max_output_tokens": 800, "temperature": 0.2, "top_p": 1, "top_k": 32}
        self.text_config = {"max_output_tokens": 512, "temperature": 0.0, "top_p": 1}
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]

        self.conv = []
        self.model = self.setup()
        self.video_folder_path = model_dict['video_folder_path']
        self.retriever = VideoRetriever(model_dict['video_folder_path'],
                                        metadata_path =model_dict['metadata_path'])

    def setup(self,):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        vision_model = GenerativeModel("gemini-pro-vision")
        return vision_model
    
    def reinit(self,):
        self.conv = []
    
    def save_gemini_conv(self, role, contents):
        system_promt = ''
        if role == 'system':
            system_promt = contents
        elif role == 'user':
            if isinstance(contents, list):
                self.conv.append({'role': 'user', 'parts': contents})
            else:
                self.conv.append({'role': 'user', 'parts': [contents]})
        elif role == 'assistant':
            self.conv.append({'role': 'model', 'parts': [contents]})
    
        if system_promt:
            self.conv[0]['parts'].insert(0, f"*{system_promt}*") # @
        
    def gen_caption(self, video="demo_video.mp4"):
        video_uri = upload_blob(video, video)
        video_file = Part.from_uri(video_uri, mime_type="video/mp4")

        contents = [video_file, 'Generate a caption for this video.']
        response_2 = self.model.generate_content(contents, generation_config=self.vision_config) #, safety_settings = self.safety_settings
        description_video = response_2.text
        return description_video
    
    def query_one(self, system_msg, retrieval_query, va_query):
        self.save_gemini_conv('system', system_msg)

        video_info = self.retriever.find_video(retrieval_query)
        video_path = self.video_folder_path + '/' + video_info['video_path'].split('/')[-1]

        video_uri = upload_blob(video_path, video_path.split('/')[-1])
        video_file = Part.from_uri(video_uri, mime_type="video/mp4")

        contents = [video_file, va_query]
        self.save_gemini_conv('user', contents) # 대화가 저장되고 대화 기반으로 contents 구성
        # contents = self.conv
        response = self.model.generate_content(contents, generation_config=self.vision_config) #, safety_settings = self.safety_settings
        behavior_rationale = response.text
        return behavior_rationale

from video_rag.utils import get_similar_video_from_query, get_video_document_metadata
class VideoRetriever:
    def __init__(self, 
                 video_folder_path,
                 video_description_prompt=None,
                 use_video_description=False,
                 metadata_path=None):
        self.multimodal_model = self.setup()
        self.video_metadata_df = get_video_document_metadata(
            self.multimodal_model,  # we are passing gemini 1.0 pro vision model
            video_folder_path=video_folder_path,
            video_description_prompt=video_description_prompt,
            embedding_size=1408,
            metadata_path=metadata_path,
            use_video_description=use_video_description,
        )

    def setup(self,):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH
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