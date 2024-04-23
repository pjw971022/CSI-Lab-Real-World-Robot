
import openai
from time import sleep
from openai import OpenAI
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from utils import load_prompt, DynamicObservation, IterableDynamicObservation
import time
from LLM_cache import DiskCache

import google.generativeai as genai
genai.configure(api_key='AIzaSyDRv4MkxqaTD9Nn4xDieqFkHbf8Ny4eU_I')
MAX_TRAIAL = 5
WORKSPACE='/home/pjw971022/workspace'
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
            }
        ]
from physical_reasoning.vqa_utils import call_meta_chat
class LMP:
    """Language Model Program (LMP), adopted from Code as Policies."""
    def __init__(self, name, cfg, fixed_vars, variable_vars, debug=False, env='rlbench',tracker=None):
        self._name = name
        self._cfg = cfg
        self._debug = debug
        try:
            self._base_prompt = load_prompt(f"{env}/{self._cfg['prompt_fname']}.txt")
        except FileNotFoundError:
            import ipdb;ipdb.set_trace()
        self._stop_tokens = list(self._cfg['stop'])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''
        self._context = None
        self._cache = DiskCache(load_cache=self._cfg['load_cache'])
        self.tracker = tracker
        self.text_config = {"max_output_tokens": self._cfg['max_tokens'], "temperature": self._cfg['temperature'], "top_p": 1, "stop_sequences" : self._stop_tokens}
        self.safety_settings = SAFETY_SETTINGS
        
    def clear_exec_hist(self):
        self.exec_hist = ''

    def build_prompt(self, query):
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = f"from utils import {', '.join(self._variable_vars.keys())}"
        else:
            variable_vars_imports_str = ''
        prompt = self._base_prompt.replace('{variable_vars_imports}', variable_vars_imports_str)

        if self._cfg['maintain_session'] and self.exec_hist != '':
            prompt += f'\n{self.exec_hist}'
        
        prompt += '\n'  # separate prompted examples with the query part

        if self._cfg['include_context']:
            assert self._context is not None, 'context is None'
            prompt += f'\n{self._context}'

        user_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f'\n{user_query}'
        return prompt, user_query
    
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

    def _cached_api_call(self, **kwargs):
        # add special prompt for chat endpoint
        user1 = kwargs.pop('prompt')
        new_query = '# Query:' + user1.split('# Query:')[-1]
        user1 = ''.join(user1.split('# Query:')[:-1]).strip()
        user1 = f"I would like you to help me write Python code to control a robot arm operating in a tabletop environment. Please complete the code every time when I give you new query. Pay attention to appeared patterns in the given context code. Be thorough and thoughtful in your code. Do not include any import statement. Do not repeat my question. Do not provide any text explanation (comment in code is okay). I will first give you the context of the code below:\n\n```\n{user1}\n```\n\nNote that x is back to front, y is left to right, and z is bottom to up."
        assistant1 = f'Got it. I will complete what you give me next.'
        user2 = new_query

        ############# Code Error fix ############
        if self._name == 'planner':
            user1 += " Please plan very specifically using all the given context."
        elif self._name == 'composer':
            user1 += " When you need to grasp an object, use the code movable=parse_query_obj('gripper')"
        elif self._name == 'parse_query_obj':
            user1 += " Objects obtained using the detect function only have attributes position and aabb. For example, object.position"
        elif 'map' in self._name:
            user1 += ' The object retrieved using parse_query_obj has only the attributes position and aabb. For example, object.position, object.aabb.'

        # handle given context (this was written originally for completion endpoint)
        if user1.split('\n')[-4].startswith('objects = ['):
            obj_context = user1.split('\n')[-4]
            # remove obj_context from user1
            user1 = '\n'.join(user1.split('\n')[:-4]) + '\n' + '\n'.join(user1.split('\n')[-3:])
            # add obj_context to user2
            user2 = obj_context.strip() + '\n' + user2
        messages=[
            {"role": "system", "content": "You are a helpful assistant that pays attention to the user's instructions and writes good python code for operating a robot arm in a tabletop environment."},
            {"role": "user", "content": user1},
            {"role": "assistant", "content": assistant1},
            {"role": "user", "content": user2},
        ]
        kwargs['messages'] = messages
    
        # check whether completion endpoint or chat endpoint is used
        if any([chat_model in kwargs['model'] for chat_model in ['llama3-70b','llama3-8b']]):
            meta_messages = self.transform_to_llama(messages)
            ret = call_meta_chat(meta_messages,
                        model = kwargs['model'],
                temperature = self._cfg['temperature'],
                max_tokens_in_use = self._cfg['max_tokens'])
            return ret
        
        elif  any([chat_model in kwargs['model'] for chat_model in ['gemini-pro', 'gemini-1.0-pro-latest', 'gemini-pro-vision']]):
            model = genai.GenerativeModel(kwargs['model'])

            messages = self.transform_to_gemini(messages)
            kwargs['messages'] = messages
            response = model.generate_content(messages,
                                generation_config = self.text_config,
                                safety_settings=self.safety_settings)
            ret = response.text
            # post processing
            ret = ret.replace('```', '').replace('python', '').strip()
            self._cache[kwargs] = ret
            return ret
        
        elif kwargs['model'] != 'gpt-3.5-turbo-instruct' and \
            any([chat_model in kwargs['model'] for chat_model in ['gpt-3.5', 'gpt-4','gpt-4-1106-vision-preview']]):
           
            if kwargs in self._cache:
                print('(using cache)', end=' ')
                return self._cache[kwargs], None
            else:
                client = OpenAI()
                response = client.chat.completions.create(**kwargs)
                ret = response.choices[0].message.content.strip()
                usg = response.usage
                # post processing
                ret = ret.replace('```', '').replace('python', '').strip()
                self._cache[kwargs] = ret
                return ret, usg
        else:
            if kwargs in self._cache:
                print('(using cache)', end=' ')
                return self._cache[kwargs], None
            else:
                client = OpenAI()
                response = client.chat.completions.create(**kwargs)
                ret = response.choices[0].message.content.strip()
                usg = response.usage
                # ret = openai.Completion.create(**kwargs)['choices'][0]['text']
                self._cache[kwargs] = ret
                return ret, usg

    def __call__(self, query, **kwargs):
        prompt, user_query = self.build_prompt(query)
        start_time = time.time()
        try_cnt = 0
        if (self._name == 'planner') and ('plan_code' in kwargs.keys()):
            code_str = kwargs['plan_code']
        else:
            while MAX_TRAIAL > try_cnt:
                try_cnt+=1
                if any([chat_model in self._cfg['model'] for chat_model in ['llama3-70b','llama3-8b']]):
                    try:
                        code_str = self._cached_api_call(
                            prompt=prompt,
                            stop=self._stop_tokens,
                            temperature=self._cfg['temperature'],
                            model=self._cfg['model'],
                            max_tokens=self._cfg['max_tokens']
                        )
                        print(f'*** Llama3 API call took {time.time() - start_time:.2f}s ***')
                        break
                    except Exception as e:
                        print(f'Llama3 API got err {e}')
                        print('Retrying after 0.1s.')
                        sleep(0.1)
                if any([chat_model in self._cfg['model'] for chat_model in ['gpt-3.5', 'gpt-4','gpt-4-1106-preview']]):
                    try:
                        code_str, usage = self._cached_api_call(
                            prompt=prompt,
                            stop=self._stop_tokens,
                            temperature=self._cfg['temperature'],
                            model=self._cfg['model'],
                            max_tokens=self._cfg['max_tokens']
                        )
                        # print out usage
                        cost = self.tracker.calculate_cost(usage)
                        total_cost = self.tracker.get_total_cost()

                        print(f'cost = {cost}, total-cost: {total_cost}')
                        print(f'*** OpenAI API call took {time.time() - start_time:.2f}s ***')
                        break
                    except (openai.RateLimitError, openai.APIConnectionError) as e:
                        print(f'OpenAI API got err {e}')
                        print('Retrying after 3s.')
                        sleep(3)
                elif any([chat_model in self._cfg['model'] for chat_model in ['gemini-pro', 'gemini-1.0-pro-latest','gemini-pro-vision']]):
                    try:
                        code_str = self._cached_api_call(
                            prompt=prompt,
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
                        sleep(3)
                else:
                    raise NotImplementedError

        if self._cfg['include_context']:
            assert self._context is not None, 'context is None'
            to_exec = f'{self._context}\n{code_str}'
            to_log = f'{self._context}\n{user_query}\n{code_str}'
        else:
            to_exec = code_str
            to_log = f'{user_query}\n{to_exec}'
        
        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())

        if self._cfg['include_context']:
            print('#'*40 + f'\n## "{self._name}" generated code\n' + f'## context: "{self._context}"\n' + '#'*40 + f'\n{to_log_pretty}\n')
        else:
            print('#'*40 + f'\n## "{self._name}" generated code\n' + '#'*40 + f'\n{to_log_pretty}\n')

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        # return function instead of executing it so we can replan using latest obs（do not do this for high-level UIs)
        if not self._name in ['composer', 'planner']:
            to_exec = 'def ret_val():\n' + to_exec.replace('ret_val = ', 'return ')
            to_exec = to_exec.replace('\n', '\n    ')

        if self._debug:
            # only "execute" function performs actions in environment, so we comment it out
            action_str = ['execute(']
            try:
                for s in action_str:
                    exec_safe(to_exec.replace(s, f'# {s}'), gvars, lvars)
            except Exception as e:
                print(f'Error: {e}')
                import pdb ; pdb.set_trace()
        else:
            exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f'\n{to_log.strip()}'

        if self._cfg['maintain_session']:
            self._variable_vars.update(lvars)

        if self._cfg['has_return']:
            if self._name == 'parse_query_obj':
                try:
                    # there may be multiple objects returned, but we also want them to be unevaluated functions so that we can access latest obs
                    return IterableDynamicObservation(lvars[self._cfg['return_val_name']])
                except AssertionError:
                    return DynamicObservation(lvars[self._cfg['return_val_name']])
            return lvars[self._cfg['return_val_name']]


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
    try:
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        print(f'Error executing code:\n{code_str}')
        raise e

import re
def extract_strings(text):
    # Define a regular expression pattern to find all occurrences of strings
    # enclosed within ~~ inside the composer function calls
    pattern = re.compile(r'composer\("([^"]*)"\)')    
    # Find all matches in the text
    matches = pattern.findall(text)
    
    return matches