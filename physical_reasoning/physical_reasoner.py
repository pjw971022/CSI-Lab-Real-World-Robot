import json
import sys
from PIL import Image

from physical_reasoning.vqa_utils import call_openai_chat, call_google_chat, exec_safe, SpatialVLMQuery, ShapeLLMQuery
WORKSPACE = "/home/pjw971022/workspace"

sys.path.append(WORKSPACE + '/Sembot/VideoRAG/utils')
sys.path.append(WORKSPACE + '/Sembot/VideoRAG/utils/LLaVA')


import zmq
import numpy as np
import time
from PIL import Image
from physical_reasoning.pc_3d.ShapeLLM.llava.mm_utils import load_pts

class Physical_AS_REASONER(object):
    def __init__(self, settings_dict, use_server=True, debug=False):
        self.openai_settings = settings_dict["openai_settings"]
        self.google_settings = settings_dict["google_settings"]
        self.api_mode = settings_dict["api_mode"]
        self.total_cost = 0
        """generate the system message"""
        self.physical_mode = settings_dict["physical_mode"]
        self.llm_sys_msg = settings_dict["prompt"]["system"]
        self.llm_fewshot = settings_dict["prompt"]["fewshot"].rstrip('\n')
        
        if self.physical_mode == '2d':
            self.spatialVLM_prompt = settings_dict["llava_prompt"]
            self.visual_assistant = SpatialVLMQuery(settings_dict['llava_model_2d'])
        elif self.physical_mode=='3d':
            self.shapeLLM_prompt = settings_dict["llava_prompt"]
            self.visual_assistant = ShapeLLMQuery(settings_dict['llava_model_3d'])
        
        if debug:
            self.video_context_assistant = """
                The video shows a person walking into the kitchen and opening the refrigerator.
                Move to the refrigerator: The person walks towards the refrigerator with their arms at their sides.
                Grasp the refrigerator handle: They reach out and grasp the handle of the refrigerator with their right hand.
                Pull the refrigerator door open: They pull the door open with their right hand, while their left hand rests on the refrigerator door.
                Reach into the refrigerator: They reach into the refrigerator with their right hand and grab a container of food.
                Close the refrigerator door: They close the refrigerator door with their right hand.
                Move to the trash can: They walk to the trash can with the container of food in their right hand.
                Open the trash can lid: They use their left hand to open the lid of the trash can.
                Throw away the food: They throw the container of food into the trash can with their right hand.
                Close the trash can lid: They close the lid of the trash can with their left hand.
                Move away from the trash can: They walk away from the trash can with their arms at their sides.
            """
        else:
            self.video_context_assistant = VideoLLMQuery()

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
        sent_msg = True
        while True:
            if sent_msg:
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
                sent_msg = True
                continue

            self.recv_dict = json.loads(recv_data)
            if self.prev_chat is None:
                self.instr = self.recv_dict['instruction']
                self.possible_obj_list = self.recv_dict['possible_obj']
                self.possible_obj = ', '.join(self.possible_obj_list)
                self.prev_chat = f' {self.instr}\n[Possible object] {self.possible_obj}'

            if self.recv_dict['robot_res'] == '':
                self.prev_chat += f'[Friend 2] {self.recv_dict["robot_res"]}\n'
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
                    pts = self.recv_dict['pts'] #= load_pts(self.recv_dict['pts'])
                    vq_answer = self._ask_3d_visual_assistant_question_server(pts, resp)
                self.prev_chat += f'[Friend 1] {vq_answer}\n'
                sent_msg = False

            elif '[Answer]' in resp:
                data = json.dumps(resp.replace('[Answer]', ''))
                self.socket.send_string(data)
                sent_msg = True

            elif '(To Friend 2)' in resp:
                data = json.dumps(resp.replace('(To Friend 2) ', ''))
                self.socket.send_string(data)
                sent_msg = True
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
    
    def _ask_3d_visual_assistant_question_local(self,pts, question):
        llava_sys_msg = self.shapeLLM_prompt["system"]
        formated_question = self.shapeLLM_prompt["question_template"]
        formated_question = formated_question.replace("<question>", self.possible_obj)
        formated_question = formated_question.replace("<question>", question)
        start_time = time.time()
        answer = self.visual_assistant.query_one(pts, llava_sys_msg, formated_question)
        print(f'*** Shape VLM call took {time.time() - start_time:.2f}s ***')
        return answer
    
    def _ask_3d_visual_assistant_question_server(self, pts, question):
        llava_sys_msg = self.shapeLLM_prompt["system"]

        formated_question = self.shapeLLM_prompt["question_template"]
        formated_question = formated_question.replace("<obj_list>", self.possible_obj)
        formated_question = formated_question.replace("<question>", question)

        start_time = time.time()
        answer = self.visual_assistant.query_one(pts, llava_sys_msg, formated_question)
        print(f'*** Shape VLM call took {time.time() - start_time:.2f}s ***')
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
