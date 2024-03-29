import os
import json
import tarfile
import io
import sys
from PIL import Image
from tqdm import tqdm
import traceback

from constants import RAW_DATA_BY_EP_DIR, EARLY_TERMINATION_TAG, DONE_TAG
from utils import call_openai_chat, call_google_chat, exec_safe
sys.path.append('/home/pjw971022/workspace/Sembot/VideoRAG/utils')
sys.path.append('/home/pjw971022/workspace/Sembot/VideoRAG/utils/LLaVA')
import zmq
import numpy as np
from LLaVA.llava.api.llava_query import LLaVAQuery, SpatialVLMQuery
import time
class SPATIAL_AS_REASONER(object):
    def __init__(self, settings_dict):
        self.openai_settings = settings_dict["openai_settings"]
        self.google_settings = settings_dict["google_settings"]
        self.api_mode = settings_dict["api_mode"]
        self.total_cost = 0

        """generate the system message"""
        self.llm_sys_msg = settings_dict["prompt"]["system"]
        self.llm_fewshot = settings_dict["prompt"]["fewshot"].rstrip('\n')
        self.spatialVLM_prompt = settings_dict["llava_prompt"]

        self.recv_dict = {}
        self.visual_assistant = SpatialVLMQuery(settings_dict['llava_model'])
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)  # REP (REPLY) socket
        self.socket.bind("tcp://*:5555")
        print("Chat Server Open")

    def generate_fg_skill(self,):
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
            # 프롬프트 디자인 추가 하고 싶은 요구사항
            # 1. 물체가 있는지에 대해서는 묻지 않는다.
            # 2. 물체의 구체적인 위치를 물을 건지 상대적인 위치를 물을 건지 나눈다.

            self.recv_dict = json.loads(recv_data)
            if prev_chat is None:
                self.instr = self.recv_dict['instruction']
                self.possible_obj_list = self.recv_dict['possible_obj']
                possible_obj = ', '.join(self.possible_obj_list)
                prev_chat = f' {self.instr}\n[Possible object] {possible_obj}'
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
                vq_answer = self._ask_visual_assistant_question('overhead_rgb', resp)
                prev_chat += f'[Friend 1] {vq_answer}\n' 
            elif '(To Friend 2)' in resp:
                data = json.dumps(resp.replace('(To Friend 2) ', ''))
                self.socket.send_string(data)
                prev_chat += '[Friend 2] Done.\n'
            else:
                raise NotImplementedError
            
    def _build_query_messages(self, prev_chat):
        messages = [{
                        "role": "system", 
                        "content": self.llm_sys_msg
                    }]
        prompt = f"{self.llm_fewshot} {prev_chat}\n[you] "
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

    def _ask_visual_assistant_question(self, image_type, question):
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
        formated_question = formated_question.replace("<question>", possible_obj)
        formated_question = formated_question.replace("<question>", question)
        start_time = time.time()
        answer = self.visual_assistant.query_one(image, llava_sys_msg, formated_question)
        print(f'*** Spatial VLM call took {time.time() - start_time:.2f}s ***')

        return answer