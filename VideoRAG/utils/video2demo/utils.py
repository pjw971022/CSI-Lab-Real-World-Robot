import os
import time

import openai
os.environ['OPENAI_API_KEY'] =  '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' 

def safe_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

# import vertexai
# from vertexai.preview.generative_models import GenerativeModel, Part
location = "asia-northeast3"
project_id = "gemini-api-415903"
key_path = "/home/pjw971022/workspace/Sembot/sembot/src/configs/gemini-api-415903-0f8224218c2c.json"
vision_config = {"max_output_tokens": 800, "temperature": 0.0, "top_p": 1, "top_k": 32}
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

# def setup(model, location, project_id):
#     os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
#     vertexai.init(project=project_id, location=location)
#     vision_model = GenerativeModel(model)
#     return vision_model
    
# def call_vertexai_chat(messages, model="gemini-pro", temperature=0.0, max_tokens_in_use=2048):
#     text_config = {"max_output_tokens": max_tokens_in_use, "temperature": temperature, "top_p": 1}
#     model = setup(model, location, project_id)

#     start_time = time.time()
#     num_attempts = 0
#     while num_attempts < 10:
#         try:
#             response = model.generate_content(
#                 contents=messages,
#                 generation_config=text_config
#             )

#             print(f'*** Google API call took {time.time() - start_time:.2f}s ***')
#             return response.text, None
#         except Exception as e:
#             print(f'Google API got err {e}')
#             print('Retrying after 3s.')
#             time.sleep(3)

import google.generativeai as genai
genai.configure(api_key='AIzaSyDRv4MkxqaTD9Nn4xDieqFkHbf8Ny4eU_I')

def call_google_chat(messages, model="gemini-pro", temperature=0.0, max_tokens_in_use=2048):
    text_config = {"max_output_tokens": max_tokens_in_use, "temperature": temperature, "top_p": 1}
    safety_settings = SAFETY_SETTINGS
    start_time = time.time()
    num_attempts = 0
    model = genai.GenerativeModel(model)
    while num_attempts < 10:
        try:
            response = model.generate_content(
                contents=messages,
                generation_config=text_config, safety_settings = safety_settings
            )
            parts = response.parts
            generated_sequence = ''
            for part in parts:
                generated_sequence += part.text
            print(f'*** Google API call took {time.time() - start_time:.2f}s ***')
            return generated_sequence
        except Exception as e:
            print(f'Google API got err {e}')
            print('Retrying after 3s.')
            time.sleep(3)

from openai import OpenAI
def call_openai_chat(messages, model="gpt-4", temperature=0.0, max_tokens_in_use=2048):
    """
    Sends a request with a chat conversation to OpenAI's chat API and returns a response.

    Parameters:
        messages (list)
            A list of dictionaries containing the messages to send to the chatbot.
        model (str)
            The model to use for the chatbot. Default is "gpt-3.5-turbo".
        temperature (float)
            The temperature to use for the chatbot. Defaults to 0. Note that a temperature
            of 0 does not guarantee the same response (https://platform.openai.com/docs/models/gpt-3-5).
    
    Returns:
        response (Optional[dict])
            The response from OpenAI's chat API, if any.
    """
    num_attempts = 0
    while num_attempts < 10:
        try:

            client = OpenAI()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens_in_use
            )
            ret = response.choices[0].message.content.strip()
            usg = response.usage
            return ret, usg
    
            # return response['choices'][0]['message']['content'].strip(), response['usage']
        except openai.AuthenticationError as e:
            print(e)
            return None
        except openai.RateLimitError as e:
            print(e)
            print("Sleeping for 10 seconds...")
            time.sleep(10)
            num_attempts += 1
        except openai.BadRequestError as e: # ServiceUnavailableError
            print(e)
            print("Sleeping for 10 seconds...")
            time.sleep(10)
            num_attempts += 1
        except openai.InternalServerError as e:
            print(e)
            print(f"Trying with max-token={max_tokens_in_use} to max-token={max_tokens_in_use - 100}")
            max_tokens_in_use -= 100
        except Exception as e:
            print(e)
            print("other bugs, try again in 5 seconds...")
            time.sleep(5)


def merge_dicts(dicts):
    return {
        k : v 
        for d in dicts
        for k, v in d.items()
    }

def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ['import', '__']
    for phrase in banned_phrases:
        assert phrase not in code_str
  
    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    
    exec(code_str, custom_gvars, lvars)