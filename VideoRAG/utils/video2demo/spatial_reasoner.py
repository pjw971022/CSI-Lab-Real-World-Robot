import json
import sys
from PIL import Image
from constants import RAW_DATA_BY_EP_DIR, EARLY_TERMINATION_TAG, DONE_TAG
from utils import call_openai_chat, call_google_chat, exec_safe

import zmq
import numpy as np
import time
import json
import torch
import os
import math

from PIL import Image
# from typing import List, Dict, Optional


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

import io
import base64
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

def image_to_base64_data_uri(image_input):
    # Check if the input is a file path (string)
    if isinstance(image_input, str):
        with open(image_input, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode('utf-8')

    # Check if the input is a PIL Image
    elif isinstance(image_input, Image.Image):
        buffer = io.BytesIO()
        image_input.save(buffer, format="PNG")  # You can change the format if needed
        base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

    else:
        raise ValueError("Unsupported input type. Input must be a file path or a PIL.Image.Image instance.")

    return f"data:image/png;base64,{base64_data}"

class SpatialVLMQuery():
    def __init__(self, model_dict) -> None:
        self.chat_handler = Llava15ChatHandler(clip_model_path=model_dict['mmproj'], verbose=False)
        self.spacellava = Llama(model_path=model_dict['model_path'], chat_handler=self.chat_handler, n_ctx=2048, logits_all=True, n_gpu_layers = 1,verbose=False)

    def query_one(self, image: Image.Image, system_msg: str, question: str, return_conv: bool = False) -> str:
        return self.query_conv(image, system_msg, question)
    
    def query_conv(self, image: Image.Image, system_msg: str, question) -> str:
        data_uri = image_to_base64_data_uri(image)
        messages = [
            {"role": "system", "content":system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type" : "text", "text": question}
                    ]
                }
            ]
        results = self.spacellava.create_chat_completion(messages=messages, temperature=0.2)
        return results["choices"][0]["message"]["content"]


class SPATIAL_AS_REASONER(object):
    def __init__(self, settings_dict, use_server=True):
        self.openai_settings = settings_dict["openai_settings"]
        self.google_settings = settings_dict["google_settings"]
        self.api_mode = settings_dict["api_mode"]
        self.total_cost = 0
        """generate the system message"""
        self.llm_sys_msg = settings_dict["prompt"]["system"]
        self.llm_fewshot = settings_dict["prompt"]["fewshot"].rstrip('\n')
        self.spatialVLM_prompt = settings_dict["llava_prompt"]
        self.visual_assistant = SpatialVLMQuery(settings_dict['llava_model'])
        self.prev_chat = None
        if use_server:
            self.recv_dict = {}
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)  # REP (REPLY) socket
            self.socket.bind("tcp://*:5555")
            print("Chat Server Open")
            
    def generate_fg_skill_server(self,):
        print("Start Chat LLM/VLM/API")
        vq_answer = None
        prev_chat = None
        while True:
            try:
                recv_data = self.socket.recv_string()
            except:
                import ipdb; ipdb.set_trace()
            if recv_data=='reset':
                print("######### Reset All Chat #########")
                vq_answer = None
                prev_chat = None
                data = json.dumps('complete reset')
                self.socket.send_string(data)
                continue
            self.recv_dict = json.loads(recv_data)
            if prev_chat is None:
                self.instr = self.recv_dict['instruction']
                self.possible_obj_list = self.recv_dict['possible_obj']
                possible_obj = ', '.join(self.possible_obj_list)
                prev_chat = f' {self.instr}\n[Possible object] {possible_obj}'
            if self.recv_dict['robot_res'] == '':
                prev_chat += f'[Friend 2] {self.recv_dict["robot_res"]}\n'
            google_messages, openai_messages = self._build_query_messages(prev_chat)
            if self.api_mode == 'google':
                resp = call_google_chat(google_messages,
                                model = self.google_settings["model"],
                                temperature = self.google_settings["temperature"],
                                max_tokens_in_use = self.google_settings["max_tokens"])
            elif self.api_mode == 'openai':
                resp, usage = call_openai_chat(openai_messages,
                                model = self.openai_settings["model"],
                                temperature = self.openai_settings["temperature"],
                                max_tokens_in_use = self.openai_settings["max_tokens"])
                cost = 0.01 * int(usage.prompt_tokens) / 1000.0 + 0.03 * int(usage.completion_tokens) / 1000.0
                self.total_cost += cost
                print(f'cost = {cost}, total-cost: {self.total_cost}, total-tokens={usage.total_tokens}')
            resp = resp.split('\n')[0]
            prev_chat += f' {resp}\n'
            print(f'=======response=======')
            print(resp)
            # resp = '(To Friend 1) Can you see the container of sugar?'
            if '[Answer]' in resp:
                data = json.dumps(resp.replace('[Answer]', ''))
                self.socket.send_string(data)
            elif '(To Friend 1)' in resp:
                resp.replace('(To Friend 1) ', '')
                vq_answer = self._ask_visual_assistant_question_server('overhead_rgb', resp)
                prev_chat += f'[Friend 1] {vq_answer}\n'
            elif '(To Friend 2)' in resp:
                data = json.dumps(resp.replace('(To Friend 2) ', ''))
                self.socket.send_string(data)
                # prev_chat += '[Friend 2] Done.\n'
            else:
                raise NotImplementedError
    def generate_fg_skill_local(self, recv_dict):
        if recv_dict['robot_res']!='':
            self.prev_chat += f'[Friend 2] {recv_dict["robot_res"]}\n'
        if self.prev_chat is None:
            self.instr = recv_dict['instruction']
            self.possible_obj_list = recv_dict['possible_obj']
            self.possible_obj = ', '.join(self.possible_obj_list)
            self.prev_chat = f' {self.instr}\n[Possible object] {self.possible_obj}'
        google_messages, openai_messages = self._build_query_messages(self.prev_chat)
        if self.api_mode == 'google':
            resp = call_google_chat(google_messages,
                            model = self.google_settings["model"],
                            temperature = self.google_settings["temperature"],
                            max_tokens_in_use = self.google_settings["max_tokens"])
        elif self.api_mode == 'openai':
            resp, usage = call_openai_chat(openai_messages,
                            model = self.openai_settings["model"],
                            temperature = self.openai_settings["temperature"],
                            max_tokens_in_use = self.openai_settings["max_tokens"])
            cost = 0.01 * int(usage.prompt_tokens) / 1000.0 + 0.03 * int(usage.completion_tokens) / 1000.0
            self.total_cost += cost
            print(f'cost = {cost}, total-cost: {self.total_cost}, total-tokens={usage.total_tokens}')
        resp = resp.split('\n')[0]
        self.prev_chat += f' {resp}\n'
        print(f'=======response=======')
        print(resp)
        if '[Answer]' in resp:
            return resp
        elif '(To Friend 1)' in resp:
            resp.replace('(To Friend 1) ', '')
            image = recv_dict['wrist_rgb']
            vq_answer = self._ask_visual_assistant_question_local( image, resp)
            self.prev_chat += f'[Friend 1] {vq_answer}\n'
        elif '(To Friend 2)' in resp:
            return resp
        else:
            raise NotImplementedError
        
    def _build_query_messages(self):
        messages = [{
                        "role": "system",
                        "content": self.llm_sys_msg
                    }]
        prompt = f"{self.llm_fewshot} {self.prev_chat}\n[you] "
        messages.append({
                            "role": "user",
                            "content": prompt
                        })
        google_messages = self.transform_to_gemini(messages)
        return google_messages, messages
    def transform_to_gemini(self, messages_chatgpt):
        messages_gemini = []
        system_promt = ''
        for message in messages_chatgpt:
            if message['role'] == 'system':
                system_promt = message['content']
            elif message['role'] == 'user':
                messages_gemini.append({'role': 'user', 'parts': [message['content']]})
            elif message['role'] == 'assistant':
                messages_gemini.append({'role': 'model', 'parts': [message['content']]})
        if system_promt:
            messages_gemini[0]['parts'].insert(0, f"*{system_promt}*")
        return messages_gemini
    def _ask_visual_assistant_question_local(self, image, question):
        llava_sys_msg = self.spatialVLM_prompt["system"]
        formated_question = self.spatialVLM_prompt["question_template"]
        formated_question = formated_question.replace("<question>", self.possible_obj)
        formated_question = formated_question.replace("<question>", question)
        start_time = time.time()
        answer = self.visual_assistant.query_one(image, llava_sys_msg, formated_question)
        print(f'*** Spatial VLM call took {time.time() - start_time:.2f}s ***')
        return answer
    def _ask_visual_assistant_question_server(self, image_type, question):
        llava_sys_msg = self.spatialVLM_prompt["system"]
        # make sure that the image type is one of keys in recv_dict: front_rgb / wrist_rgb / overhead_rgb / left_shoulder_rgb / right_shoulder_rgb / instruction
        for _image_type in self.recv_dict.keys():
            if  'rgb' not in _image_type :
                continue
            image_array = np.array(self.recv_dict[_image_type]).astype(np.uint8)
            _image = Image.fromarray(image_array)
            _image.save(f'rgb_obs/{_image_type}.jpg')
            if _image_type == image_type:
                image = _image
        formated_question = self.spatialVLM_prompt["question_template"]
        formated_question = formated_question.replace("<question>", self.possible_obj)
        formated_question = formated_question.replace("<question>", question)
        start_time = time.time()
        answer = self.visual_assistant.query_one(image, llava_sys_msg, formated_question)
        print(f'*** Spatial VLM call took {time.time() - start_time:.2f}s ***')
        return answer