import numpy as np
import re
import sys
sys.path.append('/home/jinwoo/workspace/Sembot/sembot/src/physical_reasoning')
from vqa_utils import call_openai_chat, call_google_chat
from web_tools.core.engines.google import Search as GoogleSearch
from PIL import Image

def parse_confidence(input_string):
    match = re.search(r'(\d+)%', input_string)
    if match:
        return int(match.group(1))
    return None

def parse_plan(input_string):
    parsed_plan = input_string
    return parsed_plan

def parse_question(input_string):
    return input_string

def google_search(query):
    # init a search engine
    gsearch = GoogleSearch(proxy=None)
    # will automatically parse Google and corresponding web pages
    gresults = gsearch.search(query, cache=True, page_cache=True, topk=1, end_year=2024)
    print(gresults)
    return gresults['description']

class InteractiveAgent:
    def __init__(self, composer_ui, settings_dict) -> None:
        self.composer_ui = composer_ui

        self.fewshot_prompt = settings_dict["prompt"]['fewshot']
        self.system_prompt = settings_dict["prompt"]['system']
        self.rule_prompt = settings_dict["prompt"]['rule']

        self.openai_settings = settings_dict["openai_settings"]
        self.google_settings = settings_dict["google_settings"]
        self.api_mode = settings_dict["api_mode"]
        self.chat_with = settings_dict["chat_with"]
        self.is_black_box = settings_dict["is_black_box"]

        self.total_cost = 0
        self.prev_chat = None
        self.image_path = settings_dict["image_path"]
    def __call__(self, obs_dict):
        if self.prev_chat is None:
            self.instr = obs_dict['instruction']
            self.possible_obj_list = obs_dict['possible_obj']
            self.possible_obj = ', '.join(self.possible_obj_list)
            self.init_obs_desc_ = self.obs_captioning()
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
        print(f'======= LLM output =======')
        print(resp)

        if self.is_black_box:
            plan_resp = resp.split('\n')[-1]
            conf_resp = resp.split('\n')[-2]
            question_resp = resp.split('\n')[-3]
            confidence = parse_confidence(conf_resp)
        else:
            confidence = None   # TODO: calculate token logprob

        #### if confidence > 80%
        if confidence > 80:
            plan = parse_plan(plan_resp)
            feedback = self.execute_plan(plan) # success or (fail & reason)
            self.prev_chat += f'[Feedback] {feedback}\n'
        else:
            question = parse_question(question_resp)
            answer = self.execute_question(question)
            self.prev_chat += f'[Answer] {answer}\n'

    def execute_plan(self, plan):
        print(f'======= Execute Plan =======')
        _feedback = self.composer_ui(plan)
        import ipdb;ipdb.set_trace()
        feedback = self.obs_captioning(plan, feedback=True)
        return feedback

    def execute_question(self, question):
        print(f'======= Question =======')
        print(f'{question}')
        if self.chat_with == 'human':
            answer = input(f'Answer:')
        elif self.chat_with == 'google':
            answer = google_search(question)
        print(f'======= Response =======')
        print(answer)
        self.prev_chat += f'[Answer] {answer}\n'
        return answer
    
    def _build_query_messages(self):
        messages = [{
                        "role": "system",
                        "content": self.system_prompt
                    }]        
        prompt = f"{self.rule_prompt}\n[Environment] {self.init_obs_desc_}\n "
         
        
        if self.fewshot_prompt:
            prompt += f"{self.fewshot_prompt}\n"
        prompt =  f"{self.prev_chat}\n[Plan] "
        messages.append({
                            "role": "user",
                            "content": prompt
                        })
        google_messages = self.transform_to_gemini(messages)
        return google_messages, messages
    
    def obs_captioning(self, plan=None, feedback=False):
        if feedback:
            text_prompt = f"Based on the images, please let me know if '{plan}' was successful or if it failed, explain why it failed."
        else:
            text_prompt = f"These images capture the same scenario from multiple views. [possible objects] {self.possible_obj}. Please summarize the situation centered around the robot and [possible objects] on the table as depicted in the images."

        front_image = Image.open(self.image_path + 'front.png')
        left_shoulder_image = Image.open(self.image_path + 'left_shoulder.png')
        right_shoulder_image = Image.open(self.image_path + 'right_shoulder.png')
        overhead_image = Image.open(self.image_path + 'overhead.png')
        wrist_image = Image.open(self.image_path + 'wrist.png')

        messages = [front_image, left_shoulder_image, right_shoulder_image, overhead_image, wrist_image, text_prompt]
        resp = call_google_chat(messages=messages,
                        model = 'gemini-1.0-pro-vision',
                        temperature = 0.2,
                        max_tokens_in_use = 2049)
        return resp
    
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