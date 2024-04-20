import os
import time

import openai
os.environ['OPENAI_API_KEY'] =  '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' 

def safe_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


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
    empty_fn = lambda *model_dict, **kwmodel_dict: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    
    exec(code_str, custom_gvars, lvars)


from PIL import Image
import io
import base64




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
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        self.chat_handler = Llava15ChatHandler(clip_model_path=model_dict['mmproj'], verbose=False)
        self.spacellava = Llama(model_path=model_dict['model_path'], chat_handler=self.chat_handler, n_ctx=2048, logits_all=True, n_gpu_layers = 1, verbose=False)

    def reinit(self,):
        pass

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


from pc_3d.ShapeLLM.llava.utils import disable_torch_init
from pc_3d.ShapeLLM.llava.model.builder import load_pretrained_model
from pc_3d.ShapeLLM.llava.conversation import conv_templates, SeparatorStyle
from pc_3d.ShapeLLM.llava.constants import POINT_TOKEN_INDEX, DEFAULT_POINT_TOKEN, DEFAULT_PT_START_TOKEN, DEFAULT_PT_END_TOKEN
from pc_3d.ShapeLLM.llava.mm_utils import load_pts, process_pts, rotation, tokenizer_point_token, get_model_name_from_path, \
    KeywordsStoppingCriteria
import torch
from transformers import TextStreamer

class ShapeLLMQuery(object):
    def __init__(self, model_dict) -> None:
        disable_torch_init()
        model_name = get_model_name_from_path(model_dict['model_path'])
        self.tokenizer, self.model, self.context_len = load_pretrained_model(model_dict['model_path'], model_dict['model_base'], model_name, model_dict['load_8bit'],
                                                            model_dict['load_4bit'], device=model_dict['device'])

        conv_mode = "llava_v1"
        if model_dict['conv_mode'] is not None and conv_mode != model_dict['conv_mode']:
            print(
                '[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode,
                                                                                                                model_dict['conv_mode'],
                                                                                                                model_dict['conv_mode']))
        else:
            model_dict['conv_mode'] = conv_mode

        self.conv = conv_templates[model_dict['conv_mode']].copy()

        self.model_dict = model_dict
        self.temperature = model_dict['temperature']
        self.max_new_tokens = model_dict['max_new_tokens']
        self.objaverse = False

    def reinit(self,):
        self.conv = conv_templates[self.model_dict['conv_mode']].copy()
    
    def query_one(self, pts, system_msg: str, question: str, debug=False):
        self.conv.system = system_msg
        if self.objaverse:
            pts[:, :3] = rotation(pts[:, :3], [0, 0, -90])
        pts_tensor = process_pts(pts, self.model.config).unsqueeze(0)
        pts_tensor = pts_tensor.to(self.model.device, dtype=torch.float16)

        inp = question
        if pts is not None:
            # first message
            if self.model.config.mm_use_pt_start_end:
                inp = DEFAULT_PT_START_TOKEN + DEFAULT_POINT_TOKEN + DEFAULT_PT_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_POINT_TOKEN + '\n' + inp
            self.conv.append_message(self.conv.roles[0], inp)
            pts = None
        else:
            # later messages
            self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()

        input_ids = tokenizer_point_token(prompt, self.tokenizer, POINT_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                points=pts_tensor,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                # streamer=streamer,
                use_cache=True,
                # stopping_criteria=[stopping_criteria]
                )

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        self.conv.messages[-1][-1] = outputs

        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
