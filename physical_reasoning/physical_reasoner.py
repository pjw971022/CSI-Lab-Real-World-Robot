import json
import sys
from PIL import Image

from vqa_utils import call_openai_chat, call_google_chat, exec_safe, SpatialVLMQuery, ShapeLLMQuery
from video_utils import VideoLLMQuery
from pc_3d.ShapeLLM.llava.mm_utils import load_pts
WORKSPACE = "/home/pjw971022/workspace"

sys.path.append(WORKSPACE + '/Sembot/VideoRAG/utils')
sys.path.append(WORKSPACE + '/Sembot/VideoRAG/utils/LLaVA')


import zmq
import numpy as np
import time
from PIL import Image

class Physical_AS_REASONER(object):
    def __init__(self, settings_dict, use_server=True, debug=False):
        self.openai_settings = settings_dict["openai_settings"]
        self.google_settings = settings_dict["google_settings"]
        self.api_mode = settings_dict["api_mode"]
        self.total_cost = 0
        """generate the system message"""
        self.physical_mode = settings_dict["physical_mode"]
        self.llm_sys_msg = settings_dict["llm_prompt"]["system"]
        self.llm_fewshot = settings_dict["llm_prompt"]["fewshot"].rstrip('\n')
        
        if self.physical_mode == '2d':
            self.spatialVLM_prompt = settings_dict["llava_prompt_2d"]
            self.visual_assistant = SpatialVLMQuery(settings_dict['llava_model_2d'])
        elif self.physical_mode=='3d':
            self.shapeLLM_prompt = settings_dict["llava_prompt_3d"]
            self.visual_assistant = ShapeLLMQuery(settings_dict['llava_model_3d'])

        if debug:
            self.video_context_assistant = ''
        else:
            self.videoVLM_prompt = settings_dict["video_vlm_prompt"]
            self.video_context_assistant = VideoLLMQuery(
                model_dict=settings_dict["video_vlm_model"],
            )

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
        self.prev_chat = None
        while True:
            try:
                recv_data = self.socket.recv_string()
            except:
                import ipdb; ipdb.set_trace()

            if recv_data=='reset':
                print("######### Reset All Chat #########")
                vq_answer = None
                self.prev_chat = None
                data = json.dumps('complete reset')
                self.socket.send_string(data)
                continue
            else:
                self.recv_dict = json.loads(recv_data)
                if self.prev_chat is None:
                    self.instr = self.recv_dict['instruction']
                    self.possible_obj_list = self.recv_dict['possible_obj']
                    self.task_name = self.recv_dict['task_name']
                    self.possible_obj = ', '.join(self.possible_obj_list)
                    self.prev_chat = f"[Question] Transform '{self.instr}' into a 'fine-grained instruction' suitable for the environment.\n[Possible object] {self.possible_obj}"
                while True:
                    google_messages, openai_messages = self._build_query_messages()
                    
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

                    if '(To Friend 1)' in resp:
                        resp.replace('(To Friend 1) ', '')
                        if self.physical_mode == '2d':
                            vq_answer = self._ask_2d_visual_assistant_question_server('wrist_rgb', resp)
                        elif self.physical_mode == '3d':
                            pts_file = WORKSPACE + f'/Sembot/physical_reasoning/pc_3d/pcd_data/{self.task_name}.ply'
                            pts = load_pts(pts_file)
                            vq_answer = self._ask_3d_visual_assistant_question_server(pts, resp)
                        self.prev_chat += f'[Friend 1] {vq_answer}\n'
                    elif '(To Friend 2)' in resp:
                        vq_answer = self._ask_video_assistant_question(resp)
                        self.prev_chat += f'[Friend 2] {vq_answer}\n' 
                    elif '[Answer]' in resp:
                        data = json.dumps(resp.replace('[Answer]', ''))
                        self.socket.send_string(data)
                        break
                    else:
                        raise NotImplementedError
            
    def generate_fg_skill_local(self, recv_dict):
        if recv_dict['robot_res']!='':
            self.prev_chat += f'[Friend 2] {recv_dict["robot_res"]}\n'

        if self.prev_chat is None:
            self.instr = recv_dict['instruction']
            self.possible_obj_list = recv_dict['possible_obj']
            self.possible_obj = ', '.join(self.possible_obj_list)
            self.prev_chat = f"[Question] Transform '{self.instr}' into a 'fine-grained instruction' suitable for the environment.\n[Possible object] {self.possible_obj}"
        
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
            if self.physical_mode == '2d':
                image = recv_dict['wrist_rgb']
                vq_answer = self._ask_2d_visual_assistant_question_local( image, resp)
            elif self.physical_mode == '3d':
                vq_answer = self._ask_3d_visual_assistant_question_local(resp)

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
    
    def _ask_3d_visual_assistant_question_local(self, pts, question):
        llava_sys_msg = self.shapeLLM_prompt["system"]
        formated_question = self.shapeLLM_prompt["question_template"]
        formated_question = formated_question.replace("<question>", self.possible_obj)
        formated_question = formated_question.replace("<question>", question)
        start_time = time.time()
        answer = self.visual_assistant.query_one(pts, llava_sys_msg, formated_question)
        print(f'*** Shape VLM call took {time.time() - start_time:.2f}s ***')
        return answer
    
    def _ask_3d_visual_assistant_question_server(self, pts, question):

        formated_question = self.shapeLLM_prompt["question_template"]
        formated_question = formated_question.replace("<obj_list>", self.possible_obj)
        formated_question = formated_question.replace("<question>", question)

        start_time = time.time()
        answer = self.visual_assistant.query_one(pts, formated_question)
        print(f'*** Shape LLM call took {time.time() - start_time:.2f}s ***')
        return answer

    def _ask_video_assistant_question(self, question):
        video_sys_msg = self.videoVLM_prompt["system"]
        start_time = time.time()
        answer = self.video_context_assistant.query_one(question, video_sys_msg, question)
        print(f'*** Video VLM call took {time.time() - start_time:.2f}s ***')
        return answer

    def _ask_2d_visual_assistant_question_local(self, image, question):
        llava_sys_msg = self.spatialVLM_prompt["system"]
        formated_question = self.spatialVLM_prompt["question_template"]
        formated_question = formated_question.replace("<question>", self.possible_obj)
        formated_question = formated_question.replace("<question>", question)
        start_time = time.time()
        answer = self.visual_assistant.query_one(image, llava_sys_msg, formated_question)
        print(f'*** Spatial VLM call took {time.time() - start_time:.2f}s ***')
        return answer
    
    def _ask_2d_visual_assistant_question_server(self, image_type, question):
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
        formated_question = formated_question.replace("<obj_list>", self.possible_obj)
        formated_question = formated_question.replace("<question>", question)
        start_time = time.time()
        answer = self.visual_assistant.query_one(image, llava_sys_msg, formated_question)
        print(f'*** Spatial VLM call took {time.time() - start_time:.2f}s ***')
        return answer
