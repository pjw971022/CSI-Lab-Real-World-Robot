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
sys.path.append('/home/pjw971022/workspace/Sembot/sembot/src/VideoRAG/utils')
sys.path.append('/home/pjw971022/workspace/Sembot/sembot/src/VideoRAG/utils/LLaVA')

from LLaVA.llava.api.llava_query import LLaVAQuery

class GPT_AS_REASONER(object):
    def __init__(self, ep_id, settings_dict, output_folder_path, full_restart):
        self.ep_id = ep_id

        self.chat_log_size = settings_dict["query_build_settings"]["chat_log_size"]
        self.round_size = settings_dict["query_build_settings"]["round_size"]
        self.prev_pred_size = settings_dict["query_build_settings"]["prev_prediction_size"]
        self.prev_pred_keys_to_use = settings_dict["query_build_settings"]["prev_prediction_keys_to_use"]

        self.llava_prompt_data = settings_dict["llava_prompt"]
        self.query_template = settings_dict["query_template"]
        self.openai_settings = settings_dict["openai_settings"]
        self.google_settings = settings_dict["google_settings"]
        self.api_mode = settings_dict["api_mode"]
        self.total_cost = 0

        """generate the system message"""
        self.sys_msg = settings_dict["prompt"]["system"]
        if "domain" in settings_dict["prompt"]:
            # replace domain
            self.sys_msg = self.sys_msg.replace("<domain>", settings_dict["prompt"]["domain"])
        self.sys_msg = self.sys_msg.strip()

        """Load the visual assistant"""
        self.visual_assistant = LLaVAQuery(model_path = settings_dict["llava_model"]["model_path"],
                                           model_base = settings_dict["llava_model"]["model_base"],
                                            conv_mode = settings_dict["llava_model"]["conv_mode"],
                                            temp = settings_dict["llava_model"]["temperature"],
                                            )

        """Loading all necessary data"""
        # load the raw data json
        with open(os.path.join(RAW_DATA_BY_EP_DIR, f"{ep_id}_data.json"), "r") as fin:
            self.raw_data = json.load(fin)[ep_id]

        # sort the timestep just in case
        self.timestep_list = [int(k.split("_")[-1]) for k in list(self.raw_data.keys())]
        self.timestep_list.sort()

        # load the images ahead of time
        images_path_dict = {
            "start_image": [],
            "random_image": [],
            "end_image": []
        }

        for i in range(len(self.timestep_list)):
            timestep = self.timestep_list[i]

            frame_data = self.raw_data[f"{ep_id}_{timestep}"]

            if i == 0:
                # assuming that all the images for this episode are stored in this tar file
                tar_path = frame_data["tar_path"]
            
            images_path_dict['start_image'].append(frame_data['start_image'])
            images_path_dict['random_image'].append(frame_data['random_image'])
            images_path_dict['end_image'].append(frame_data['end_image'])

        self.images_dict = {
            "start_image": [],
            "random_image": [],
            "end_image": []
        }

        with tarfile.open(tar_path) as tf:
            for image_type in images_path_dict:
                tarinfos = [tf.getmember(image_file) for image_file in images_path_dict[image_type]]
                images = [tf.extractfile(tarinfo) for tarinfo in tarinfos]
                images = [image.read() for image in images]
                self.images_dict[image_type] = [Image.open(io.BytesIO(image)).convert('RGB') for image in images]
                
        """load the progress or create the json for the first time"""
        self.output_filepath = os.path.join(output_folder_path, f"{settings_dict['prompt_name']}_{settings_dict['yaml_version']}_{ep_id}_output.json")

        if os.path.exists(self.output_filepath) and not full_restart:
            # load it from last time
            self._load_result_json()
        else:
            if os.path.exists(self.output_filepath):
                print(f"Warning: there's already a result json file at {self.output_filepath}")
                if input("Are you sure you want to start from scratch? (y/n)").strip() != "y":
                    sys.exit()
            self.results_json = {}

            # pre-save before starting
            self._save_result_json()

    def generate(self):
        """
        Main function to call. Generate state-action predication for all timesteps in a video
        """
        # loop through each action interval
        for i in (pbar := tqdm(range(len(self.timestep_list)))):
            pbar.set_description(f"{self.ep_id}: {self.timestep_list[i]}")

            # if self.timestep_list[i] >= 39 and self.timestep_list[i] <= 71:
            self.curr_idx = i
            self.round_num, self.had_generated_reason_and_act = self._init_or_load_progress(self.curr_idx)

            self._predict_one_state_action()

        print(f"Total Cost (not include previously predicted): {self.total_cost}")

    def _predict_one_state_action(self):
        """
        Generate state-action for one timestep in the video
        """
        prev_pred_list = self._get_previous_prediction(self.curr_idx)

        timestep = self.timestep_list[self.curr_idx]
        timestep_id = f"{self.ep_id}_{timestep}"

        while not self._can_terminate():
            if not self.had_generated_reason_and_act:
                reason, action = self._generate_reason_and_act(prev_pred_list)
                self.had_generated_reason_and_act = True
            else:
                # we have already generated reason and act, but we need to load it
                reason, action = self._get_reason_and_act(self.curr_idx, self.round_num)

            print(f"========== {self.ep_id} Round {self.round_num} ==========")
            print(f"reason:\n{reason}")
            print(f"act:\n{action}")

            # have not excuted act yet
            if self.results_json[timestep_id]["chat_log"][-1]["obs"] == "":
                obs = self._execute_action(action)
                # update the json to save the latest observation
                self.results_json[timestep_id]["chat_log"][-1]["obs"] = obs
                self._save_result_json()

                print(f"obs:\n{obs}")

            self.round_num += 1
            self.had_generated_reason_and_act = False

        if not self._has_states_prediction():
            # GPT must make a prediction on states right now
            reason, action = self._generate_reason_and_act(prev_pred_list, must_predict_states=True)

            self._execute_action(action, must_predict_states=True)

        if not self._has_prediction():
            # GPT must make a prediction on action right now
            reason, action = self._generate_reason_and_act(prev_pred_list, must_predict_action=True)

            self._execute_action(action, must_predict_action=True)

    def _generate_reason_and_act(self, prev_pred_list, must_predict_states=False, must_predict_action=False):
        timestep = self.timestep_list[self.curr_idx]
        timestep_id = f"{self.ep_id}_{timestep}"

        google_messages, openai_messages = self._build_query_messages(timestep_id, prev_pred_list, must_predict_states, must_predict_action)

        # for debugging
        if self.round_num == 1 or self.round_num % 3 == 0 or self.round_num == self.round_size:
            for x in openai_messages:
                if (self.round_num == 1 or self.round_num == self.round_size)and self.curr_idx == 0:
                    # also print out system message
                    print(f'======={x["role"]}')
                    print(x['content'])
                elif x["role"] != "system":
                    print(f'======={x["role"]}')
                    content_to_print = x['content']
                    if "[[Current Instruction]]" in content_to_print:
                        content_to_print, _, _= content_to_print.partition("[[Current Instruction]]")
                        content_to_print = content_to_print.strip().strip("\n").strip()
                    print(content_to_print)

        for x in openai_messages:
            if x["role"] != "system":
                print(f'======={x["role"]}')
                content_to_print = x['content']
                if "[[Current Instruction]]" in content_to_print:
                    content_to_print, _, _= content_to_print.partition("[[Current Instruction]]")
                    content_to_print = content_to_print.strip().strip("\n").strip()
                print(content_to_print)

        for x in openai_messages:
            print(f'======={x["role"]}')
            print(x['content'])
        input("check messages")
        
        # call openai/google api
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

            # print out usage
            cost = 0.01 * int(usage["prompt_tokens"]) / 1000.0 + 0.03 * int(usage["completion_tokens"]) / 1000.0
            self.total_cost += cost
            print(f'cost = {cost}, total-cost: {self.total_cost}, total-tokens={usage["total_tokens"]}')

        # clean up the response
        if "\\n" in resp:
            resp = resp.replace("\\n", "\n")

        try:
            resp_as_json = json.loads(resp, strict=False)
            assert "reason" in resp_as_json
            assert "act" in resp_as_json
        except Exception:
            # TODO: eventually we can make it more robust, asking openai to regenerate (but cap it)
            print(resp)
            traceback.print_exc()
            print("Error: unable to load response")
            sys.exit()

        # save this round of conversation (have not executed "act" yet, so obs is empty)
        self.results_json[timestep_id]["chat_log"].append({
            "reason": resp_as_json["reason"],
            "act": resp_as_json["act"],
            "obs": ""
        })

        self._save_result_json()

        return resp_as_json["reason"], resp_as_json["act"]
    

    def _build_query_messages(self, timestep_id, prev_pred_list, must_predict_states=False, must_predict_action=False):
        messages = [{
                        "role": "system", 
                        "content": self.sys_msg
                    }]
        
        if must_predict_action:
            curr_status = self.query_template["final_main"]
        else:
            curr_status = self.query_template["main"]

        # add the previous prediciton
        if prev_pred_list == []:
            prev_pred_str = "None"
        else:
            prev_pred_str = json.dumps(prev_pred_list, indent=1).strip()
        curr_status = curr_status.replace("<previous_to_fill>", prev_pred_str)

        # add the latest self.chat_log_size of conversation
        chat_log_list = self.results_json[timestep_id]["chat_log"][-self.chat_log_size:]

        if chat_log_list == []:
            chat_log_str = "None"
        else:
            chat_log_str = ""
            for i in range(len(chat_log_list)):
                chat_data = chat_log_list[i]

                if type(chat_data["act"]) == str and "finish" in chat_data["act"]:
                    question = "I want to finish asking question now."
                    image_type = "None"
                else:
                    if type(chat_data["act"]) != dict:
                        act_dict = json.loads(chat_data["act"])
                    else:
                        act_dict = chat_data["act"]

                    if "start_states" in act_dict or "end_states" in act_dict:
                        continue

                    question = act_dict["question"]
                    image_type = act_dict["image_to_ask_about"]
                
                chat_str = self.query_template["chat_log_template"].replace("<question>", question.strip())
                chat_str = chat_str.replace("<answer>", chat_data["obs"].strip())
                chat_str = chat_str.replace("<image_type>", image_type)
                if i == 0:
                    # replace the first obs to be from system
                    chat_str = chat_str.replace("VA:", "SYS:", 1)
                chat_str = chat_str.strip()

                chat_log_str = f"{chat_log_str}{chat_str}\n"
            
            chat_log_str = chat_log_str.strip()

        curr_status = curr_status.replace("<chat_log_to_fill>", chat_log_str)

        # load the proper instruction on what gpt can do
        if must_predict_states:
            curr_status = curr_status.replace("<action_space_to_fill>", self.query_template['states_prediction_instruction'].strip())
        elif must_predict_action:
            curr_status = curr_status.replace("<action_space_to_fill>", self.query_template['action_prediction_instruction'].strip())
            curr_status = curr_status.replace("<obj_list>", str(self._get_clean_objects(self.raw_data[timestep_id]['objects'])).strip())
            curr_status = curr_status.replace("<states_prediction_to_fill>", f"start_states: {str(self.results_json[timestep_id]['prediction']['start_states'])}\nend_states: {str(self.results_json[timestep_id]['prediction']['end_states'])}")
        else:
            curr_status = curr_status.replace("<action_space_to_fill>", self.query_template['q_and_a_instruction'].strip())
            curr_status = curr_status.replace("<rounds_left>", str(self.round_size - self.round_num)) 

        messages.append({
                            "role": "user",
                            "content": curr_status.strip()
                        })
        # OpenAI api > Google api
        # if self.api_mode == 'google':
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

    def _execute_action(self, action, must_predict_states=False, must_predict_action=False):
        """
        Execute the action if it's valid
        """
        if must_predict_states or must_predict_action:
            # make sure it's not asking question or finishing asking question
            assert "question" not in action

            # we are not executing the code now. We assume that "action" is just the prediction dict
            self._make_prediction(action)
        elif type(action) == str and "finish" in action:
            self._finish_asking_question()
        else:
            # ask visual assitant question
            try:
                if type(action) != dict:
                    action_dict = json.loads(action)
                else:
                    action_dict = action
            except:
                print(action)
                # TODO: better error handling
                input("Error: unable to convert questions to a dictionary")
            
            assert "image_to_ask_about" in action_dict
            assert "question" in action_dict

            self._ask_visual_assistant_question(action_dict["image_to_ask_about"], action["question"])

        # get the return value (a legacy from before when we are executing the code generated by gpt)
        return_val = self.return_val

        if EARLY_TERMINATION_TAG in return_val or DONE_TAG in return_val:
            obs = ""
        else:
            obs = return_val.strip().strip("\n").strip()

        # flush return val
        self.return_val = ""

        return obs
   
    """========================================================================================

    Actions GPT can choose to do

    ========================================================================================"""
    def _ask_visual_assistant_question(self, image_type, question):
        # make sure that the image type is one of keys in images_dict: "start_image", "random_image", "end_image"
        assert image_type in self.images_dict

        # flush what we have before as return value
        self.return_val = ""

        llava_sys_msg = self.llava_prompt_data["system"]
        image = self.images_dict[image_type][self.curr_idx]

        formated_question = self.llava_prompt_data["question_template"]
        timestep_id = f"{self.ep_id}_{self.timestep_list[self.curr_idx]}"
        formated_question = formated_question.replace("<obj_list>", str(self._get_clean_objects(self.raw_data[timestep_id]["objects"])))
        formated_question = formated_question.replace("<question>", question)

        answer = self.visual_assistant.query_one(image, llava_sys_msg, formated_question)
        
        # a hack to get return value
        self.return_val = answer

    def _finish_asking_question(self):
        # flush what we have before as return value
        self.return_val = ""

        # set the current round number to round size
        #   so that it will trigger early termination
        self.round_num = self.round_size + 1

        # a hack to get return value
        self.return_val = EARLY_TERMINATION_TAG


    def _make_prediction(self, pred):
        # flush what we have before as return value
        self.return_val = ""

        if type(pred) == str:
            try:
                pred_dict = json.loads(pred)
            except:
                print(pred)
                # TODO: better error handling
                input("Error: unable to convert prediction to a dictionary")
        else:
            pred_dict = pred

        print(f"prediction:\n{json.dumps(pred_dict, indent=2)}")

        assert "start_states" in pred_dict or "end_states" in pred_dict or "action" in pred_dict
        
        timestep_id = f"{self.ep_id}_{self.timestep_list[self.curr_idx]}"

        # add the object list
        pred_dict["objects"] = self.raw_data[timestep_id]["objects"]

        if "start_states" in pred_dict:
            self.results_json[timestep_id]["prediction"]["start_states"] = pred_dict["start_states"]

        if "end_states" in pred_dict:
            self.results_json[timestep_id]["prediction"]["end_states"] = pred_dict["end_states"]

        if "action" in pred_dict:
            self.results_json[timestep_id]["prediction"]["action"] = pred_dict["action"]

        self._save_result_json()

        # a hack to get return value
        self.return_val = DONE_TAG

    """========================================================================================

    Helper functions to get various information

    ========================================================================================"""
    def _get_previous_prediction(self, idx):
        prev_pred_list = []

        if idx > 0:
            # get the latest self.prev_pred_size # of past predicitons
            for j in range(max(0, idx - self.prev_pred_size), idx):
                timestep_id = f"{self.ep_id}_{self.timestep_list[j]}"

                try: 
                    pred = self.results_json[timestep_id]["prediction"]

                    # there might be some keys from the prediction
                    #   that we don't want to include in the prompt
                    filtered_pred = {k: pred[k] for k in pred if k in self.prev_pred_keys_to_use}

                    prev_pred_list.append(filtered_pred)
                except:
                    continue
            
        return prev_pred_list


    def _get_init_obs(self, timestep_id):
        obj_list = self._get_clean_objects(self.raw_data[timestep_id]["objects"])

        # get the observation template and fill in the obs number
        init_obs= self.query_template["observation_template"].replace("<obs_num>", "0")
        init_obs = init_obs.strip()
        # fill init obs with the initial obs message
        init_obs = init_obs.replace("<obs_to_fill>", self.query_template["init_obs_template"])
        init_obs = init_obs.replace("<obj_list>", str(obj_list))

        init_obs = init_obs.strip()

        return init_obs


    def _get_reason_and_act(self, idx, round_num):
        timestep_id = f"{self.ep_id}_{self.timestep_list[idx]}"

        chat_log = self.results_json[timestep_id]["chat_log"]

        return chat_log[round_num]["reason"], chat_log[round_num]["act"]


    def _get_clean_objects(self, obj_list):
        clean_obj_list = [o.strip() for o in obj_list]

        for hand_name in ["left hand", "right hand"]:
            if hand_name in clean_obj_list:
                clean_obj_list = [o.replace(hand_name, "hand") for o in clean_obj_list]

        return list(set(clean_obj_list))


    def _can_terminate(self):
        """
        Return true if the gpt4 has reached its termination condition
            - it ran out of rounds to ask questions
            - it has prediction already
        """
        return (self.round_num > self.round_size) or self._has_states_prediction()
    

    def _has_prediction(self):
        timestep_id = f"{self.ep_id}_{self.timestep_list[self.curr_idx]}"

        curr_pred_result = self.results_json[timestep_id]["prediction"]

        if curr_pred_result == {}:
            return False
        
        has_start_states = "start_states" in curr_pred_result and len(curr_pred_result["start_states"]) > 0
        has_end_states = "end_states" in curr_pred_result and len(curr_pred_result["end_states"]) > 0
        has_action = "action" in curr_pred_result and type(curr_pred_result["action"]) == str and curr_pred_result["action"] != ""

        return has_start_states and has_end_states and has_action

    def _has_states_prediction(self):
        timestep_id = f"{self.ep_id}_{self.timestep_list[self.curr_idx]}"

        curr_pred_result = self.results_json[timestep_id]["prediction"]

        if curr_pred_result == {}:
            return False
        
        has_start_states = "start_states" in curr_pred_result and len(curr_pred_result["start_states"]) > 0
        has_end_states = "end_states" in curr_pred_result and len(curr_pred_result["end_states"]) > 0

        return has_start_states and has_end_states


    def _init_or_load_progress(self, idx):
        timestep_id = f"{self.ep_id}_{self.timestep_list[idx]}"

        if timestep_id not in self.results_json:
            # have no progress on this timestep
            #   need to initialize everything
            self.results_json[timestep_id] = {
                "prediction": {
                    "start_states": [],
                    "end_states": [],
                    "action": ""
                },
                "chat_log": [
                    {
                        "reason": f"{self.query_template['init_reason'].strip()}",
                        "act": self.query_template['init_act'],
                        "obs": f"{self.query_template['init_obs_template'].replace('<obj_list>', str(self._get_clean_objects(self.raw_data[timestep_id]['objects']))).strip()}"
                    }
                ],
                'meta_data': self.raw_data[timestep_id]
            }

            self._save_result_json()

            round_num = 1
            had_generated_reason_and_act = False
        elif self._has_states_prediction():
            # time to predict action
            round_num = self.round_size + 1

            had_generated_reason_and_act = False
        else:
            round_num = len(self.results_json[timestep_id]["chat_log"])

            if self.results_json[timestep_id]["chat_log"][-1]["obs"] == "":
                # gpt4 has generated the reason and act for round_num - 1, but it hasn't executed the act yet
                if "finish_asking_question" in self.results_json[timestep_id]["chat_log"][-1]["act"]:
                    # already decided to terminate
                    round_num = self.round_size + 1
                else:
                    round_num -= 1
                had_generated_reason_and_act = True
            else:
                had_generated_reason_and_act = False

        return round_num, had_generated_reason_and_act

    def _save_result_json(self):
         with open(self.output_filepath, "w") as fout:
            fout.write(json.dumps(self.results_json, indent=4))

    def _load_result_json(self):
        with open(self.output_filepath, "r") as fin:
            self.results_json = json.load(fin)