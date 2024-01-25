overwrite_cache = True
if overwrite_cache:
  LLM_CACHE = {}

import openai
openai.api_key = "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"
from openai import OpenAI

# from lamorel import Caller, lamorel_init
# from lamorel.server.llms import HF_LLM
import pandas as pd

# lamorel_init()

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

# from lamorel.server.llms.module_functions import BaseModuleFunction, LogScoringModuleFn
import torch
import google.generativeai as palm 
palm.configure(api_key='AIzaSyDRv4MkxqaTD9Nn4xDieqFkHbf8Ny4eU_I')
class LLMAgent:
    def __init__(self, lamorel_args, llm_type=None) -> None:
        use_cpu = False
        self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        # if llm_type =='open':
        #     self.lm_model = HF_LLM(lamorel_args.llm_args, [0], use_cpu)
        #     custom_module_functions = {}
        #     custom_module_functions["__score"] = LogScoringModuleFn(self.lm_model.pad_token, lamorel_args.llm_args.model_type,
        #                                                             lamorel_args.llm_args.pre_encode_inputs)
        #     for k, _fn in custom_module_functions.items():
        #         assert isinstance(_fn, BaseModuleFunction)
        #         _fn.device = self.lm_model.device
        #         _fn.llm_config = self.lm_model.get_model_config()
                
        #         _fn.initialize()
        #     self.lm_model.register_module_functions(custom_module_functions)
        #     self.module_function_keys = ["__score"]
        #     self.batch_size = lamorel_args.llm_args.batch_size
        #     self.num_final_beams = 3
        #     self.decoding_type = lamorel_args.decoding_type

    def palm_new_scoring(self, context, options):
        models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
        model = models[0].name
        completion = palm.generate_text(
            model=model,
            prompt=context,
            temperature=0,
            # The maximum length of the response
            max_output_tokens=800,
        )
        generated_sequence = completion.result
        generated_sequence = generated_sequence.split('.')[0]
        gen_embedding = self.sentence_model.encode(generated_sequence, convert_to_tensor=True,show_progress_bar=False)
        act_embeddings = [ self.sentence_model.encode(action, convert_to_tensor=True,show_progress_bar=False) for action in options]
        _, scores = process_action2(gen_embedding, act_embeddings)
        
        llm_scores = {action: score for action, score in zip(options, scores)}
        return llm_scores, generated_sequence
    
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