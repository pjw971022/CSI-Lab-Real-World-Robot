import numpy as np
import re
import sys
sys.path.append('/home/jinwoo/workspace/Sembot/sembot/src/physical_reasoning')
from vqa_utils import call_openai_chat, call_google_chat, call_meta_chat
from web_tools.core.engines.google import Search as GoogleSearch
from PIL import Image
from rlbench_qa_prompt import get_task_planner_prompt

def extract_between_tags(text, tag):
    """ Extract content between specified start and end tags from the given text. """
    pattern = f'<{tag}>(.*?)</{tag}>'
    # Find all non-greedy matches within the tags
    content = re.findall(pattern, text, re.DOTALL)
    if tag == 'F':
        content = content[0].replace('%', '').strip()
        content = int(content)
    else:  
        content = content[0].strip()
    return content

def google_search(query, cache, topk, end_year):
    # init a search engine
    gsearch = GoogleSearch(proxy=None)
    # will automatically parse Google and corresponding web pages
    gresults = gsearch.search(query, cache=cache, page_cache=True, topk=topk, end_year=end_year)
    return gresults['description']

import time
import pprint
def google(query, cache=True, topk=1, end_year=None, verbose=False):
    assert topk >= 1

    gresults = {"page": None, "title": None}

    trial = 0
    while gresults['page'] is None and trial < 3:
        trial += 1
        if trial > 1:
            print("Search Fail, Try again...")
        gresults = google_search(query, cache=cache, topk=topk, end_year=end_year)
        time.sleep(3 * trial)

    if verbose:
        pprint.pprint(gresults)

    return gresults

class InteractiveAgent:
    def __init__(self, composer_ui, settings_dict) -> None:
        self.composer_ui = composer_ui

        # self.fewshot_prompt = settings_dict["prompt"]['fewshot']
        # self.system_prompt = settings_dict["prompt"]['system']
        # self.rule_prompt = settings_dict["prompt"]['rule']

        self.openai_settings = settings_dict["openai_settings"]
        self.google_settings = settings_dict["google_settings"]
        self.meta_settings = settings_dict["meta_settings"]
        
        self.api_mode = settings_dict["api_mode"]
        self.chat_with = settings_dict["chat_with"]
        self.is_black_box = settings_dict["is_black_box"]
        self.confidence_threshold = settings_dict["confidence_threshold"]
        self.total_cost = 0
        # self.prev_chat = None
        self.image_path = settings_dict["image_path"]
        self.env_context_mode = settings_dict["env_context_mode"]
        self.qa_history = ''
        self.execution_history = ''
        self.feedback_history = ''

    def __call__(self, obs_dict):
        self.instr = obs_dict['instruction']
        self.possible_obj_list = obs_dict['possible_obj']
        self.possible_obj = ', '.join(self.possible_obj_list)
        while True:
            meta_messages, google_messages, openai_messages = self._build_query_messages()
            if self.api_mode == 'google':
                resp = call_google_chat(google_messages,
                                model = self.google_settings["model"],
                                temperature = self.google_settings["temperature"],
                                max_tokens_in_use = self.google_settings["max_tokens"])
                
            elif self.api_mode == 'meta':
                resp = call_meta_chat(meta_messages,
                                        model = self.meta_settings["model"],
                                temperature = self.meta_settings["temperature"],
                                max_tokens_in_use = self.meta_settings["max_tokens"])
                
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
                confidence = extract_between_tags(resp, 'F')
            else:
                confidence = None   # TODO: calculate token logprob

            #### if confidence > 80%
            plan = extract_between_tags(resp, 'P')
            if plan == 'done.':
                break
            if int(confidence) >= self.confidence_threshold * 100:
                feedback = self.execute_plan(plan) # success or (fail & reason)
                self.feedback_history += feedback

            else:
                question = extract_between_tags(resp, 'Q')
                answer = self.execute_question(question)
                self.qa_history += f'Q: {question} A: {answer}\n'

    def execute_plan(self, plan):
        print(f'======= Execute Plan =======')
        # import ipdb;ipdb.set_trace()
        self.composer_ui(plan)
        feedback = self.obs_captioning(plan, feedback=True)
        return feedback

    def execute_question(self, question):
        print(f'======= Question =======')
        print(f'{question}')
        if self.chat_with == 'human':
            answer = input(f'Answer:')
            if answer == 'q':
                exit()
        elif self.chat_with == 'google':
            answer = google_search(question)
        print(f'======= Response =======')
        print(answer)
        return answer
    
    def _build_query_messages(self):
        prompt = get_task_planner_prompt(self.instr, self.possible_obj, execution_history=self.execution_history, qa=self.qa_history, feedback=self.feedback_history)
        messages = [{
                        "role": "user",
                        "content": prompt
                    }]
        
        google_messages = self.transform_to_gemini(messages)
        meta_messages = self.transform_to_llama(messages)
        return meta_messages, google_messages, messages
    
    def obs_captioning(self, plan=None, feedback=False):
        if self.env_context_mode == 'human' and feedback:
            resp = input('Please provide the environment context.\nINPUT:')
        else:
            if feedback:
                text_prompt = f"Based on the images, please let me know if '{plan}' was successful or if it failed, explain why it failed."
            else:
                text_prompt = f"These images capture the same scenario from multiple views. [possible objects] {self.possible_obj}. Please summarize the situation centered around the robot and [possible objects] on the table as depicted in the images."

            front_image = Image.open(self.image_path + 'front_rgb.png')
            left_shoulder_image = Image.open(self.image_path + 'left_shoulder_rgb.png')
            right_shoulder_image = Image.open(self.image_path + 'right_shoulder_rgb.png')
            overhead_image = Image.open(self.image_path + 'overhead_rgb.png')
            wrist_image = Image.open(self.image_path + 'wrist_rgb.png')

            messages = [front_image, left_shoulder_image, right_shoulder_image, overhead_image, wrist_image, text_prompt]
            resp = call_google_chat(messages=messages,
                            model = 'gemini-pro-vision',
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
    
    def transform_to_llama(self, messages_chatgpt):
        parts = []
        system_prompt = None

        for message in messages_chatgpt:
            if message['role'] == 'system':
                system_prompt = f"'systemPrompt': '{message['content']}'"
            elif message['role'] == 'user':
                parts.append(f"'user': '{message['content']}'")
            elif message['role'] == 'assistant':
                parts.append(f"'Assistant': '{message['content']}'")
        
        if system_prompt:
            prompt_structure = ",\n".join([system_prompt] + parts)
        else:
            prompt_structure = ",\n".join(parts)
        
        prompt = f"\"\"\"{{\n{prompt_structure}\n}}\"\"\""

        return prompt