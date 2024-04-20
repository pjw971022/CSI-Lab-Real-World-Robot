overwrite_cache = True
if overwrite_cache:
  LLM_CACHE = {}

OPENAI_API = "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"
from openai import OpenAI
import openai

import pandas as pd

def gpt3_call(model="gpt-3.5-turbo-instruct", prompt="", max_tokens=128, temperature=0,
              logprobs=1, echo=False):
  client = OpenAI()
  full_query = ""
  for p in prompt:
    full_query += p
  id = tuple((model, full_query, max_tokens, temperature, logprobs, echo))
  if id in LLM_CACHE.keys():
    print('cache hit, returning')
    response = LLM_CACHE[id]
  else:
    response = client.completions.create(model=model,
                                        prompt=prompt,
                                        max_tokens=max_tokens,
                                        logprobs=logprobs,
                                        )
    LLM_CACHE[id] = response
  return response

def process_action1(generated_action, admissible_actions):
    """matches LM generated action to all admissible actions
        and outputs the best matching action"""    
    def editDistance(str1, str2, m, n):
        # Create a table to store results of sub-problems
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

        # Fill d[][] in bottom up manner
        for i in range(m + 1):
            for j in range(n + 1):

                # If first string is empty, only option is to
                # insert all characters of second string
                if i == 0:
                    dp[i][j] = j  # Min. operations = j

                # If second string is empty, only option is to
                # remove all characters of second string
                elif j == 0:
                    dp[i][j] = i  # Min. operations = i

                # If last characters are same, ignore last char
                # and recur for remaining string
                elif str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]

                # If last character are different, consider all
                # possibilities and find minimum
                else:
                    dp[i][j] = 1 + min(dp[i][j - 1],  # Insert
                                       dp[i - 1][j],  # Remove
                                       dp[i - 1][j - 1])  # Replace

        return dp[m][n]

    output_action = ''
    min_edit_dist_action = 100
    dist_list = []
    for action in admissible_actions:
        dist = editDistance(str1=generated_action, str2=action,
                            m=len(generated_action), n=len(action))
        dist_list.append(dist)
        if dist < min_edit_dist_action:
            output_action = action
            min_edit_dist_action = dist

    return output_action, dist_list

from sentence_transformers import SentenceTransformer, util
def process_action2(gen_embedding, act_embeddings):
    dist_list = []
    for act_embedding in act_embeddings:
        dist = util.pytorch_cos_sim(gen_embedding, act_embedding)
        dist_list.append(dist)
    return None, dist_list

import base64
import requests
from io import BytesIO

def encode_image(image_object):
    buffered = BytesIO()
    image_object.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# from lamorel.server.llms.module_functions import BaseModuleFunction, LogScoringModuleFn
import re
import google.generativeai as genai
genai.configure(api_key='AIzaSyDRv4MkxqaTD9Nn4xDieqFkHbf8Ny4eU_I')
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from google.cloud import storage
import os
import time
WORKSPACE = "/home/jinwoo/workspace"

location = "asia-northeast3"
project_id = "gemini-video-0403"
key_path = WORKSPACE + "/Sembot/physical_reasoning/gemini-video-0403-0476b10bf020.json"
video_path = "/home/pjw971022/Sembot/real_bot/save_vision/obs/"

class LLMAgent:

    def __init__(self,) -> None:
        self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.vision_config = {"max_output_tokens": 1024, "temperature": 0.0, "top_p": 1, "top_k": 32}
        self.text_config = {"max_output_tokens": 1024, "temperature": 0.0, "top_p": 1}
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        self.model = self.setup(location, project_id)

    def setup(self, location, project_id):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        vertexai.init(project=project_id, location=location)
        vision_model = GenerativeModel("gemini-pro-vision")
        return vision_model

    def upload_blob(self, source_file_name, destination_blob_name, bucket_name = 'expert_video_demo'):
        storage_client = storage.Client(project='gemini-api-413603')
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        blob.make_public()
        return f"gs://{bucket_name}/{destination_blob_name}"

    def gpt4_generate_context(self, context, img):
        base64_image = encode_image(img)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API}"
        }
        
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": context
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
                ]
            }
            ],
            "max_tokens": 1000
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        content = response.json()['choices'][0]['message']['content']
        return content
    
    def gemini_generate_context(self, context, img):
        start_time = time.time()
        while True:
            try:
                model = genai.GenerativeModel('gemini-pro-vision')
                response = model.generate_content(
                    contents=[context, img],
                    generation_config=self.vision_config, safety_settings = self.safety_settings
                )
            
                print(f'*** Google API call took {time.time() - start_time:.2f}s ***')
                break
            except Exception as e:
                print(f'Google API got err {e}')
                print('Retrying after 3s.')
                time.sleep(3)
        parts = response.parts
        generated_sequence = ''
        for part in parts:
            generated_sequence += part.text
        print("@@@ gen context: ", generated_sequence)
        return generated_sequence
    
    def gemini_generate_video_context(self, context, video):
        start_time = time.time()
        video_uri = self.upload_blob(video_path + video, video)
        video_file = Part.from_uri(video_uri, mime_type="video/mp4")
        contents = [video_file, context]
        while True:
            try:
                response = self.model.generate_content(contents, generation_config=self.vision_config) #, safety_settings = self.safety_settings
                print(f'*** Google API call took {time.time() - start_time:.2f}s ***')
                break
            except Exception as e:
                print(f'Google API got err {e}')
                print('Retrying after 3s.')
                time.sleep(3)
        result = response.text
        return result
    
    def gemini_gen_act(self, fewshot_prompt, planning_prompt, obs_img=None):
        start_time = time.time()
        model = genai.GenerativeModel('gemini-pro')
        while True:
            try:
                response = model.generate_content(
                    contents=fewshot_prompt + '\n' + planning_prompt,
                    generation_config = self.text_config,
                    safety_settings = self.safety_settings
                    )
                print(f'*** Google API call took {time.time() - start_time:.2f}s ***')
                break
            except Exception as e:
                print(f'Google API got err {e}')
                print('Retrying after 3s.')
                time.sleep(3)

        parts = response.parts
        generated_sequence = ''
        for part in parts:
            generated_sequence += part.text
        generated_sequence = generated_sequence.split('.')[0]
        print(f"@@@@ gen act: {generated_sequence}")
        return generated_sequence 
    
    def gemini_gen_all_plan(self, fewshot_prompt, planning_prompt):
        start_time = time.time()

        model = genai.GenerativeModel('gemini-pro')
        while True:
            try:
                response = model.generate_content(
                    contents=fewshot_prompt + '\n' + planning_prompt,
                    generation_config = self.text_config,
                    safety_settings = self.safety_settings
                    )
                print(f'*** Google API call took {time.time() - start_time:.2f}s ***')
                break    
            except Exception as e:
                print(f'Google API got err {e}')
                print('Retrying after 3s.')
                time.sleep(3)

        parts = response.parts
        generated_sequence = ''
        for part in parts:
            generated_sequence += part.text
        plan_list = re.findall(r"\[Plan \d+\].*?\.", generated_sequence)
        print(generated_sequence)
        return plan_list
    

    def gpt4_gen_all_plan(self, fewshot_prompt, planning_prompt):
        gpt_assistant_prompt = 'You are a planner of a robot arm for manipulation task.'
        message=[{"role": "assistant", "content": gpt_assistant_prompt}, {"role": "user", "content": fewshot_prompt + '\n' + planning_prompt }]
        temperature=0.0
        max_tokens=512
        frequency_penalty=0.0
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages = message,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty
        )
        generated_sequence = response.choices[0].message.content
        plan_list = re.findall(r"\[Plan \d+\].*?\.", generated_sequence)
        return plan_list

    def gpt4_gen_act(self, fewshot_prompt, planning_prompt):
        gpt_assistant_prompt = 'You are a planner of a robot arm for manipulation task'
        message=[{"role": "assistant", "content": gpt_assistant_prompt}, {"role": "user", "content": fewshot_prompt + '\n' + planning_prompt }]
        temperature=0.2
        max_tokens=256
        frequency_penalty=0.0
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages = message,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty
        )
        generated_sequence = response.choices[0].message.content
        plan = generated_sequence.split('.')[0]
        return plan

    def openllm_new_scoring(self, context, options):
        # contexts = [context + option for option in options]
        if self.decoding_type == 'greedy_token':
            generated_sequence = self.lm_model.greedy_generate(context)
            gen_embedding = self.sentence_model.encode(generated_sequence[0], convert_to_tensor=True,show_progress_bar=False)
            act_embeddings = [ self.sentence_model.encode(action, convert_to_tensor=True,show_progress_bar=False) for action in options]
            _, scores = process_action2(gen_embedding, act_embeddings)

        llm_scores = {action: score for action, score in zip(options, scores)}
        return llm_scores, generated_sequence

    
    def filter_beams(self, generated_sequences, generated_beam_scores, action_beam_scores,
                     num_final_beams):
        updated_action_beam_scores = []

        # TODO: considering batch_size = 1, update the code to adapt to bigger batch sizes
        # [num_final_beams, num_return_sequences]
        # print('\n')
        for ind, (gen_sequences, lm_scores) in enumerate(zip(generated_sequences, generated_beam_scores)):
            for gen_seq, lm_score in zip(gen_sequences, lm_scores):
                # new_context_pro = f'{prev_contexts_pro[ind]} [Step {curr_action_step}] {gen_seq}.'
                new_action_score = lm_score.item()
                new_beam_score = action_beam_scores[ind] + new_action_score
                updated_action_beam_scores.append(new_beam_score)
                
        action_beam_scores = zip(*sorted(zip(updated_action_beam_scores),
                        key=lambda x: x[0],
                        reverse=True)[:num_final_beams])  # selecting num_final_beams

        return list(action_beam_scores)


    def openllm_scoring(self, query, options):
        scores = [] 
        # print(f"@@ Prompt: {query}")
        for i in range(0, len(options), self.batch_size):
            # print(f"llama inference: {i} ...")
            batch = options[i:i+ self.batch_size]  # Get a sublist of possible actions
            result = self.lm_model.forward(module_function_keys=self.module_function_keys,contexts=[query], 
                                            candidates=[batch])
            batch_scores = [_r['__score'] for _r in result]
            scores.extend(batch_scores[0])
        # print(f"@@@ Score:{scores}")
        llm_scores = {action: score for action, score in zip(options, scores)}
        return llm_scores
    
    def gpt3_scoring(self, query, options, model="gpt-3.5-turbo-instruct", limit_num_options=None, option_start="\n", verbose=False, print_tokens=False):
        if limit_num_options:
            options = options[:limit_num_options]
        verbose and print("Scoring", len(options), "options")
        gpt3_prompt_options = [query + option for option in options]
        response = gpt3_call(
            model=model,
            prompt=gpt3_prompt_options,
            max_tokens=500,
            logprobs=1,
            temperature=0,)
        desc = "Scoring " + str(len(options)) + " options\n"

        scores = {}
        for option, choice in zip(options, response.choices):
            tokens = choice.logprobs.tokens
            token_logprobs = choice.logprobs.token_logprobs

            total_logprob = 0
            for token, token_logprob in zip(reversed(tokens), reversed(token_logprobs)):
                print_tokens and print(token, token_logprob)
                if option_start is None and not token in option:
                    break
                if token == option_start:
                    break
                total_logprob += token_logprob

            scores[option] = total_logprob 

        for i, option in enumerate(sorted(scores.items(), key=lambda x: -x[1])):
            verbose and print(option[1], "\t", option[0])
            desc = desc + str(option[1]) + "\t" + str(option[0]) + "\n"
            if i >= 10:
                break

        return scores#, response, desc