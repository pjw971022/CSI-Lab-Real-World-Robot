import numpy as np
import torch
from low_level_planner.src.utils import load_prompt, DynamicObservation, IterableDynamicObservation
from low_level_planner.src.LLM_cache import DiskCache
import google.generativeai as genai
from google.cloud import storage
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
import os
import time
from openai import OpenAI
import openai
genai.configure(api_key='AIzaSyDRv4MkxqaTD9Nn4xDieqFkHbf8Ny4eU_I')
MAX_TRAIAL = 5
SAFETY_SETTINGS = [
            {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
            },
            {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
            },
            {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
            },
            {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
            }]

def upload_blob(source_file_name, destination_blob_name, bucket_name = 'expert_video_demo'):
    storage_client = storage.Client(project='gemini-api-413603')
    bucket = storage_client.bucket(bucket_name)
    
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    
    blob.make_public()
    
    return f"gs://{bucket_name}/{destination_blob_name}"

def calculate_similarity(emb1, emb2):
    cosine_sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0)
    return cosine_sim

location = "asia-northeast3"
project_id = "gemini-api-415903"
key_path = "/home/pjw971022/workspace/Sembot/low_level_planner/src/envs/gemini-api-415903-0f8224218c2c.json"


class H_Planner:
    def __init__(self, cfg, env='rlbench'):
        self._cfg = cfg
        self._base_prompt = load_prompt(f"{env}/{self._cfg['prompt_fname']}.txt")
        self._stop_tokens = list(self._cfg['stop'])
        self._cache = DiskCache(load_cache=self._cfg['load_cache'])
        self.text_config = {"max_output_tokens": self._cfg['max_tokens'], "temperature": self._cfg['temperature'], "top_p": 1, "stop_sequences" : self._stop_tokens}
        self.safety_settings = SAFETY_SETTINGS
     
    def _cached_api_call(self, **kwargs):
        # check whether completion endpoint or chat endpoint is used
        if  any([chat_model in kwargs['model'] for chat_model in ['gemini-pro', 'gemini-pro-vision']]):
            model = genai.GenerativeModel(kwargs['model'])
            # add special prompt for chat endpoint
            system_query = "You are a helpful assistant that pays attention to the user's instructions and writes good python code for operating a robot arm in a tabletop environment."
            user1 = kwargs.pop('prompt')
            new_query = '# Query:' + user1.split('# Query:')[-1]
            user1 = ''.join(user1.split('# Query:')[:-1]).strip()
            user1 = f"I would like you to help me write Python code to control a robot arm operating in a tabletop environment. Please complete the code every time when I give you new query. Pay attention to appeared patterns in the given context code. Be thorough and thoughtful in your code. Do not include any import statement. Do not repeat my question. Do not provide any text explanation (comment in code is okay). I will first give you the context of the code below:\n\n```\n{user1}\n```\n\nNote that x is back to front, y is left to right, and z is bottom to up."
            assistant1 = f'Got it. I will complete what you give me next.'
            user2 = new_query
            # handle given context (this was written originally for completion endpoint)
            if user1.split('\n')[-4].startswith('objects = ['):
                obj_context = user1.split('\n')[-4]
                # remove obj_context from user1
                user1 = '\n'.join(user1.split('\n')[:-4]) + '\n' + '\n'.join(user1.split('\n')[-3:])
                # add obj_context to user2
                user2 = obj_context.strip() + '\n' + user2
            prompt =[{"role":"user",
                    "parts":[{"text":system_query}, {"text":user1}]},
                    {"role":"model",
                    "parts":[{"text":assistant1}]},
                    {"role":"user",
                    "parts":[{"text":user2}]}
                    ]
            kwargs['messages'] = prompt
            response = model.generate_content(prompt,
                                generation_config = self.text_config,
                                safety_settings=self.safety_settings)
            ret = response.text
            # post processing
            ret = ret.replace('```', '').replace('python', '').strip()
            self._cache[kwargs] = ret
            return ret

class L_Planner:
    def __init__(self, cfg, env='rlbench'):
        self._cfg = cfg
        self._stop_tokens = list(self._cfg['stop'])
        self._base_prompt = load_prompt(f"{env}/{self._cfg['prompt_fname']}.txt")
        self._stop_tokens = list(self._cfg['stop'])
        self._cache = DiskCache(load_cache=self._cfg['load_cache'])

        self.vision_config = {"max_output_tokens": 1024, "temperature": 0.0, "top_p": 1, "top_k": 32, "stop_sequences" : self._stop_tokens}
        self.safety_settings = SAFETY_SETTINGS

        self.model = self.setup(location, project_id)

    def build_prompt(self):
        prompt = self._base_prompt
        return prompt
    
    def setup(self, location, project_id):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        vertexai.init(project=project_id, location=location)
        vision_model = GenerativeModel("gemini-pro-vision")
        return vision_model
    
    def process_image(self, image):
        # Process the image here
        return image
    def process_query(self, query):
        # Process the query here
        return query
    
    def _preprocess(self, obs_path, query):
        image = cv2.imread(obs_path)
        processed_image = self.process_image(image)
        processed_query = self.process_query(query)
        cv2.imwrite(obs_path, processed_image)
        return obs_path, processed_query   
    
    def _cached_api_call(self, **kwargs):
        if  any([chat_model in kwargs['model'] for chat_model in ['gemini-pro', 'gemini-pro-vision']]):
            video_path = kwargs['video_path']
            fewshot_prompt = kwargs['fewshot_prompt']
            vision_obs_path, semantic_skill = self._preprocess(kwargs['vision_obs_path'], kwargs['semantic_skill'])
            extract_obs ="""
            Please provide a detailed description of the photo, focusing on the objects present, their relative positions, sizes, colors, and any notable features.
            Include information on the background and foreground elements, highlighting any interactions or relationships between objects.
            Following the instruction {}, identify relevant details that could influence the task's outcome.
            """
            extract_obs = extract_obs.format(semantic_skill)
            
            video_uri = upload_blob(video_path, video_path.split('/')[-1])
            image_uri = upload_blob(vision_obs_path, vision_obs_path.split('/')[-1])

            video_file = Part.from_uri(video_uri, mime_type="video/mp4")
            image_file = Part.from_uri(image_uri, mime_type="image/png")

            contents = [image_file, extract_obs]
            response_1 = self.model.generate_content(contents, generation_config=self.vision_config) 
            description_image = response_1.text

            contents = [video_file, f"[Current environment] {description_image}\n[Example] {fewshot_prompt}\n[Semantic skill] {semantic_skill}\nBased on the given video and the [Current environment], specify the [Semantic skill] like in the [Example]."]
            response_2 = self.model.generate_content(contents, generation_config=self.vision_config) 

            return response_2.text
    
    def __call__(self, **kwargs):
        fewshot_prompt = self.build_prompt()
        start_time = time.time()
        try_cnt = 0
        
        while MAX_TRAIAL > try_cnt:
            try_cnt+=1
            if any([chat_model in self._cfg['model'] for chat_model in ['gpt-3.5', 'gpt-4']]):
                try:
                    fine_grained_skill = self._cached_api_call(
                        fewshot_prompt=fewshot_prompt,
                        vision_obs=kwargs['vision_obs'],
                        video=kwargs['video_path'],
                        stop=self._stop_tokens,
                        temperature=self._cfg['temperature'],
                        model=self._cfg['model'],
                        max_tokens=self._cfg['max_tokens']
                    )
                    print(f'*** OpenAI API call took {time.time() - start_time:.2f}s ***')
                    break
                except (openai.RateLimitError, openai.APIConnectionError) as e:
                    print(f'OpenAI API got err {e}')
                    print('Retrying after 3s.')
                    time.sleep(3)
            elif any([chat_model in self._cfg['model'] for chat_model in ['gemini-pro', 'gemini-pro-vision']]):
                try:
                    fine_grained_skill = self._cached_api_call(
                        fewshot_prompt=fewshot_prompt,
                        vision_obs=kwargs['vision_obs'],
                        video=kwargs['video_path'],
                        stop=self._stop_tokens,
                        temperature=self._cfg['temperature'],
                        model=self._cfg['model'],
                        max_tokens=self._cfg['max_tokens']
                    )
                    time_cost = time.time() - start_time
                    print(f'*** Google API call took {time_cost:.2f}s ***')
                    break
                except Exception as e:
                    print(f'Google API got err {e}')
                    print('Retrying after 3s.')
                    time.sleep(3)
            else:
                raise NotImplementedError
        return fine_grained_skill, retrieval_query

class Retriever:
    def __init__(self):
        self.text_encoder = None
        self.video_db = Video_DB()
        self.videos = self.video_db.objects

    def retrieve_video(self, query):
        txt_emb = self.text_encoder(query)
        retrieved_video = self.find_highest_sim_video(txt_emb)
        return retrieved_video
    
    def find_highest_sim_video(self, txt_emb):
        video_embeddings = [video.embedding for video in self.videos]
        similarities = [calculate_similarity(txt_emb, emb) for emb in video_embeddings]
        highest_sim_index = similarities.index(max(similarities))
        retrieved_video = self.videos[highest_sim_index].path
        return retrieved_video

import cv2
class Video:
    def __init__(self):
        self.path = None
        self.embedding = None
        self.annotation = None

    def get_video(self):
        # Load the video using the path
        video = cv2.VideoCapture(self.path)
        return video
    
class Video_DB:
    def __init__(self):
        self.objects = []
        self.video_encoder = None
        self.load_video_embeddings('/path/to/video_embeddings.npy')
        self.load_video_annotations('/path/to/video_annotations.npy')
        
    def add_video(self, video: Video):
        self.objects.append(video)

    def load_video_embeddings(self, path):
        video_embeddings = np.load(path)
        for i, video in enumerate(self.objects):
            video.embedding = video_embeddings[i]
            self.objects[i] = video

    def load_video_annotations(self, path):
        video_annotations = np.load(path)
        for i, video in enumerate(self.objects):
            video.annotation = video_annotations[i]
            self.objects[i] = video